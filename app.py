from flask import Flask, jsonify, request, render_template, send_from_directory
import requests
import os
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs, RDConfig, Lipinski, AllChem

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='.', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

toxicophore_list = [
    {'name': 'Aromatic nitro', 'smarts': '[NX3](=O)[O-]c'},
    {'name': 'Epoxide', 'smarts': 'C1OC1'}
]

def get_uniprot_data(protein_id):
    try:
        queries = [
            f"accession:{protein_id}",
            f"gene:{protein_id}+AND+reviewed:true",
            f"{protein_id}+AND+reviewed:true"
        ]
        for query in queries:
            url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,protein_name,gene_names,organism_name,cc_function&format=json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('results'):
                result = data['results'][0]
                functions = []
                for comment in result.get('comments', []):
                    if comment['commentType'] == 'FUNCTION':
                        for text in comment.get('texts', []):
                            functions.append({
                                'description': text['value'],
                                'references': [ref['id'] for ref in text.get('references', [])]
                            })
                target_data = {
                    'accession': result['primaryAccession'],
                    'protein_name': result['proteinDescription']['recommendedName']['fullName']['value'],
                    'gene_names': result['genes'][0]['geneName']['value'] if result.get('genes') else 'N/A',
                    'organism': result['organism']['scientificName'],
                    'functions': functions
                }
                return target_data
        return None
    except Exception as e:
        logger.error(f"UniProt API Error for {protein_id}: {str(e)}")
        return None

def get_chembl_compounds(uniprot_id):
    try:
        components_url = f"https://www.ebi.ac.uk/chembl/api/data/target_component.json?accession={uniprot_id}"
        components_response = requests.get(components_url, timeout=10)
        components_response.raise_for_status()
        components_data = components_response.json()
        if not components_data.get('target_components'):
            return None

        target_ids = []
        for component in components_data['target_components']:
            if 'targets' in component:
                target_ids.extend([target['target_chembl_id'] for target in component['targets'] if 'target_chembl_id' in target])

        if not target_ids:
            return None

        compounds = []
        target_id = target_ids[0]
        activities_url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id={target_id}&standard_type=IC50&limit=100&offset=0"
        activities_response = requests.get(activities_url, timeout=10)
        activities_response.raise_for_status()
        activities_data = activities_response.json()

        for activity in activities_data.get('activities', []):
            smiles = activity.get('canonical_smiles')
            if not smiles:
                continue
            try:
                value = float(activity['standard_value']) if activity.get('standard_value') else None
                if value is None:
                    continue
                mol = Chem.MolFromSmiles(smiles)
                molecular_mass = round(Descriptors.MolWt(mol), 2) if mol else 'N/A'
                logp = round(Descriptors.MolLogP(mol), 2) if mol else 'N/A'
                AllChem.EmbedMolecule(mol)
                sdf_file = f"{activity['molecule_chembl_id']}.sdf"
                sdf_path = os.path.join(app.config['UPLOAD_FOLDER'], sdf_file)
                sdf_writer = Chem.SDWriter(sdf_path)
                sdf_writer.write(mol)
                sdf_writer.close()
                pdb_id = "N/A"
                compounds.append({
                    'chembl_id': activity['molecule_chembl_id'],
                    'smiles': smiles,
                    'sdf_file': sdf_file,
                    'pdb_id': pdb_id,
                    'type': activity['standard_type'],
                    'value': f"{value:.2f}",
                    'ic50_value': value,
                    'units': activity.get('standard_units', 'nM'),
                    'molecular_mass': molecular_mass,
                    'logp': logp
                })
            except (ValueError, KeyError) as e:
                logger.error(f"Skipping activity due to error: {str(e)}")

        if not compounds:
            return None

        unique_compounds = {c['chembl_id']: c for c in compounds}.values()
        final_compounds = list(unique_compounds)[:50]
        return final_compounds
    except requests.RequestException as e:
        logger.error(f"ChEMBL API Request Error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected Error in get_chembl_compounds: {str(e)}")
        return None

@app.route('/api/target')
def target_endpoint():
    protein_id = request.args.get('proteinId')
    if not protein_id:
        return jsonify({'error': 'Missing proteinId parameter'}), 400
    data = get_uniprot_data(protein_id)
    if data:
        return jsonify(data)
    return jsonify({'error': 'Target not found in UniProt'}), 404

@app.route('/api/chembl_leads')
def chembl_leads_endpoint():
    protein_id = request.args.get('proteinId')
    if not protein_id:
        return jsonify({'error': 'Missing proteinId parameter'}), 400
    compounds = get_chembl_compounds(protein_id)
    if compounds:
        response = {
            'count': len(compounds),
            'target': protein_id,
            'results': compounds
        }
        return jsonify(response)
    return jsonify({
        'error': 'No compounds found',
        'suggestion': 'Try verified targets like EGFR (P00533) or BRAF (P15056)',
        'chembl_query': f'https://www.ebi.ac.uk/chembl/api/data/activity.json?target_components__accession={protein_id}'
    }), 404

@app.route('/api/optimize', methods=['POST'])
def optimize_lead():
    data = request.json
    lead_smiles = data.get('leadSmiles')
    lead_ic50 = data.get('leadIc50')
    all_compounds = data.get('allCompounds', [])

    logger.debug(f"Received optimization request: leadSmiles={lead_smiles}, leadIc50={lead_ic50}, allCompounds length={len(all_compounds)}")

    try:
        if not lead_smiles or lead_ic50 is None:
            raise ValueError("Missing leadSmiles or leadIc50")

        mol = Chem.MolFromSmiles(lead_smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES string: {lead_smiles}")
        
        AllChem.EmbedMolecule(mol)
        sdf_file = "lead.sdf"
        sdf_path = os.path.join(app.config['UPLOAD_FOLDER'], sdf_file)
        sdf_writer = Chem.SDWriter(sdf_path)
        sdf_writer.write(mol)
        sdf_writer.close()

        properties = {
            'logP': round(Descriptors.MolLogP(mol), 2),
            'solubility': round(0.26 - 0.74 * Descriptors.MolLogP(mol) - 0.0068 * Descriptors.MolWt(mol), 2),
            'molecular_mass': round(Descriptors.MolWt(mol), 2),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'psa': round(Descriptors.TPSA(mol), 2),
            'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            'sdf_file': sdf_file,
            'pdb_id': 'N/A'
        }

        toxicophores = []
        for tp in toxicophore_list:
            patt = Chem.MolFromSmarts(tp['smarts'])
            if mol.HasSubstructMatch(patt):
                toxicophores.append(tp['name'])

        lipinski_violations = []
        if properties['molecular_mass'] > 500:
            lipinski_violations.append("Molecular weight > 500 Da")
        if properties['logP'] > 5:
            lipinski_violations.append("LogP > 5")
        if properties['hbd'] > 5:
            lipinski_violations.append("H-bond donors > 5")
        if properties['hba'] > 10:
            lipinski_violations.append("H-bond acceptors > 10")

        suggestions = []
        if properties['solubility'] < -5:
            suggestions.append("Low solubility (logS < -5). Add polar groups (e.g., -OH, -NH2).")
        if properties['logP'] > 5:
            suggestions.append("High LogP (> 5). Reduce alkyl chains or aromatic rings.")
        if properties['hbd'] > 5 or properties['hba'] > 10:
            suggestions.append("Excessive H-bond donors/acceptors. Simplify structure.")
        if properties['psa'] > 140:
            suggestions.append("High PSA (> 140 Å²). Reduce polar groups.")
        if toxicophores:
            suggestions.append(f"Toxic substructures: {', '.join(toxicophores)}.")
        if not suggestions:
            suggestions.append("Properties are acceptable; focus on potency.")

        lead_fp = Chem.RDKFingerprint(mol)
        similar_compounds = []
        for comp in all_compounds:
            comp_mol = Chem.MolFromSmiles(comp.get('smiles', ''))
            if not comp_mol or comp.get('ic50_value') is None:
                continue
            comp_fp = Chem.RDKFingerprint(comp_mol)
            similarity = DataStructs.FingerprintSimilarity(lead_fp, comp_fp)
            if comp['ic50_value'] < lead_ic50:
                similar_compounds.append({
                    'smiles': comp['smiles'],
                    'sdf_file': comp.get('sdf_file', 'N/A'),
                    'pdb_id': comp.get('pdb_id', 'N/A'),
                    'ic50_value': comp['ic50_value'],
                    'similarity': similarity,
                    'molecular_mass': round(Descriptors.MolWt(comp_mol), 2),
                    'logp': round(Descriptors.MolLogP(comp_mol), 2),
                    'solubility': round(0.26 - 0.74 * Descriptors.MolLogP(comp_mol) - 0.0068 * Descriptors.MolWt(comp_mol), 2),
                    'hbd': Lipinski.NumHDonors(comp_mol),
                    'hba': Lipinski.NumHAcceptors(comp_mol),
                    'psa': round(Descriptors.TPSA(comp_mol), 2)
                })

        similar_compounds.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar = similar_compounds[:5]

        response = {
            'properties': properties,
            'lipinski_violations': lipinski_violations,
            'suggestions': suggestions,
            'similarCompounds': top_similar if top_similar else [{
                'smiles': 'N/A', 'sdf_file': 'N/A', 'pdb_id': 'N/A', 'ic50_value': 'N/A', 'similarity': 0,
                'molecular_mass': 'N/A', 'logp': 'N/A', 'solubility': 'N/A',
                'hbd': 'N/A', 'hba': 'N/A', 'psa': 'N/A'
            }]
        }
        logger.debug(f"Optimization response prepared: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in optimize_lead: {str(e)}", exc_info=True)
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
<<<<<<< HEAD
    app.run(host='0.0.0.0', port=5006, debug=True)
=======
    app.run(host='0.0.0.0', port=5006, debug=True)
>>>>>>> 6ac83a8 (Initial commit)
