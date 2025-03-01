// Wait for the DOM to be fully loaded
const BASE_URL = 'http://127.0.0.1:5006';
document.addEventListener('DOMContentLoaded', function() {
    // Add header to the page
    addHeader();
    
    // Initialize fade-in animations
    initFadeAnimations();
    
    // Add hover animations to buttons
    addButtonAnimations();
    
    // Add scroll animations
    initScrollAnimations();
    
    // Enhance loading spinner
    enhanceLoadingSpinner();
    
    // Add tooltips to buttons
    addTooltips();
    
    // Add smooth scrolling
    addSmoothScrolling();
    
    // Add DataTable enhancements
    enhanceDataTables();

    // Initialize NGL viewers
    window.stage = new NGL.Stage('viewer', { backgroundColor: 'white' });
    window.stage3D = new NGL.Stage('viewer3D', { backgroundColor: 'white' });
    window.spinning = true;
    window.labelsVisible = false;

    // Bind event listeners
    bindEventListeners();
});

// Add a header to the page
function addHeader() {
    const header = document.createElement('div');
    header.className = 'header-container mb-4';
    header.innerHTML = `
        <h1 class="header-title text-center">Drug Discovery & ADMET Prediction Platform</h1>
        <nav class="header-nav text-center">
            <a href="#" class="mx-2">Home</a>
            <a href="#" class="mx-2">Documentation</a>
            <a href="#" class="mx-2">About</a>
            <a href="#" class="mx-2">Contact</a>
        </nav>
    `;
    document.body.insertBefore(header, document.body.firstChild);
}

// Initialize fade-in animations for elements as they come into view
function initFadeAnimations() {
    const fadeObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                fadeObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });
    
    const fadeElements = [
        document.getElementById('targetInfo'),
        document.getElementById('optimizationResults'),
        document.getElementById('viewer'),
        document.getElementById('viewer3D'),
        document.getElementById('results')
    ];
    
    fadeElements.forEach(el => {
        if (el) fadeObserver.observe(el);
    });
    
    const originalLoadCompounds = window.loadCompounds;
    if (originalLoadCompounds) {
        window.loadCompounds = function() {
            originalLoadCompounds.apply(this, arguments);
            const table = document.getElementById('compoundsTable');
            if (table) {
                setTimeout(() => table.classList.add('fade-in'), 300);
            }
        };
    }
}

// Add hover animations to buttons
function addButtonAnimations() {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.1)';
        });
        button.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });
}

// Initialize scroll animations
function initScrollAnimations() {
    let lastScrollTop = 0;
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const header = document.querySelector('.header-container');
        
        if (header) {
            if (scrollTop > 50) {
                header.style.padding = '0.5rem 2rem';
                header.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            } else {
                header.style.padding = '1rem 2rem';
                header.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            }
        }
        
        const revealElements = document.querySelectorAll('#targetInfo, #optimizationResults, #viewer, #viewer3D, #results');
        revealElements.forEach(el => {
            if (!el) return;
            const elementTop = el.getBoundingClientRect().top;
            const elementVisible = 150;
            if (elementTop < window.innerHeight - elementVisible) {
                el.classList.add('fade-in');
            }
        });
        
        lastScrollTop = scrollTop;
    });
}

// Enhance the loading spinner
function enhanceLoadingSpinner() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    window.showLoading = function() {
        loadingOverlay.style.display = 'flex';
    };
    window.hideLoading = function() {
        loadingOverlay.style.display = 'none';
    };
}

// Add tooltips to buttons
function addTooltips() {
    const buttons = {
        'Predict ADMET Properties': 'Predict ADMET properties for the entered compound',
        'Optimize with Gemini': 'Optimize the compound using Gemini AI',
        'Search Target': 'Search for a protein target by ID or name',
        'Load Compounds': 'Load compounds for the selected target',
        'View 3D': 'View the 3D structure of the molecule',
        'Optimize': 'Optimize the selected lead compound'
    };
    
    document.querySelectorAll('button').forEach(button => {
        const buttonText = button.textContent.trim();
        if (buttons[buttonText]) {
            button.setAttribute('title', buttons[buttonText]);
            button.addEventListener('mouseenter', function(e) {
                const tooltip = document.createElement('div');
                tooltip.className = 'tooltip';
                tooltip.textContent = this.getAttribute('title');
                tooltip.style.position = 'absolute';
                tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                tooltip.style.color = 'white';
                tooltip.style.padding = '5px 10px';
                tooltip.style.borderRadius = '4px';
                tooltip.style.fontSize = '0.8rem';
                tooltip.style.zIndex = '1000';
                tooltip.style.top = (e.target.getBoundingClientRect().bottom + 5) + 'px';
                tooltip.style.left = (e.target.getBoundingClientRect().left) + 'px';
                tooltip.style.opacity = '0';
                tooltip.style.transition = 'opacity 0.3s ease';
                document.body.appendChild(tooltip);
                setTimeout(() => tooltip.style.opacity = '1', 10);
                this.addEventListener('mouseleave', function() {
                    tooltip.style.opacity = '0';
                    setTimeout(() => document.body.removeChild(tooltip), 300);
                }, { once: true });
            });
        }
    });
}

// Add smooth scrolling
function addSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Enhance DataTables
function enhanceDataTables() {
    const checkDataTable = setInterval(() => {
        const dataTable = document.querySelector('.dataTable');
        if (dataTable) {
            clearInterval(checkDataTable);
            const tableWrapper = document.querySelector('.dataTables_wrapper');
            if (tableWrapper) tableWrapper.classList.add('table-responsive');
            const pagination = document.querySelector('.dataTables_paginate');
            if (pagination) pagination.classList.add('pagination-custom');
        }
    }, 1000);
}

// Bind event listeners
function bindEventListeners() {
    // ADMET Prediction Form
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const smiles = document.getElementById('smilesInput').value;
        showLoading();
        await fetchPrediction('/predict', smiles);
        hideLoading();
    });

    // Optimize with Gemini
    document.getElementById('geminiOptimizeBtn').addEventListener('click', async function() {
        const smiles = document.getElementById('smilesInput').value;
        showLoading();
        await fetchPrediction('/optimize_with_gemini', smiles);
        hideLoading();
    });

    // Chat with Gemini
    document.getElementById('chatForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const query = document.getElementById('chatInput').value;
        const chatMessages = document.getElementById('chatMessages');
        const userMsg = document.createElement('p');
        userMsg.innerHTML = `<strong>You:</strong> ${query}`;
        chatMessages.appendChild(userMsg);
        showLoading();
        try {
            const response = await fetch(`${BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query }),
            });
            const data = await response.json();
            if (data.error) throw new Error(data.error);
            const geminiMsg = document.createElement('p');
            geminiMsg.innerHTML = `<strong>Gemini:</strong> ${data.response}`;
            chatMessages.appendChild(geminiMsg);
        } catch (error) {
            const errorMsg = document.createElement('p');
            errorMsg.innerHTML = `<strong>Error:</strong> ${error.message}`;
            chatMessages.appendChild(errorMsg);
        }
        hideLoading();
        document.getElementById('chatInput').value = '';
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });

    // Optimize button clicks in compounds table
    $(document).on('click', '.optimize-btn', function() {
        const smiles = $(this).data('smiles');
        const ic50 = $(this).data('ic50');
        optimizeLead(smiles, ic50);
    });
}

// Utility functions
function showError(message) {
    document.getElementById('error').textContent = message;
}

function copyToClipboard(text) {
    if (!navigator.clipboard) {
        showError('Clipboard API not supported in this browser');
        return;
    }
    navigator.clipboard.writeText(text)
        .then(() => console.log('Copied:', text))
        .catch(err => {
            console.error('Copy failed:', err);
            showError('Copy failed. Please copy manually.');
        });
}

function viewInMolstar(uniprotId) {
    const url = `https://molstar.org/viewer/?uniprot=${encodeURIComponent(uniprotId)}`;
    window.open(url, '_blank');
}

// ADMET Prediction and Optimization
async function fetchPrediction(endpoint, smiles) {
    try {
        const response = await fetch(`${BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles: smiles }),
        });
        if (!response.ok) throw new Error(`Server returned status: ${response.status}`);
        const data = await response.json();
        console.log('Server Response:', data);
        if (data.error) {
            alert(data.error);
            document.getElementById('results').style.display = 'none';
            return;
        }

        document.getElementById('results').style.display = 'block';
        const propertyResults = document.getElementById('propertyResults');
        propertyResults.innerHTML = '';

        function renderProperties(variant, title) {
            const accordion = document.createElement('div');
            accordion.className = 'accordion mb-3';
            accordion.innerHTML = `
                <div class="accordion-item">
                    <h2 class="accordion-header">
                        <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#${title.toLowerCase().replace(' ', '-')}-collapse" aria-expanded="true" aria-controls="${title.toLowerCase().replace(' ', '-')}-collapse">
                            ${title}
                        </button>
                    </h2>
                    <div id="${title.toLowerCase().replace(' ', '-')}-collapse" class="accordion-collapse collapse show">
                        <div class="accordion-body">
                            <p><strong>SMILES:</strong> ${variant.smiles}</p>
                            <h4>Physicochemical Properties</h4>
                            ${Object.entries(variant.features.Physicochemical).map(([key, value]) => `<p><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}</p>`).join('')}
                            <h4>Lipophilicity</h4>
                            ${Object.entries(variant.features.Lipophilicity).map(([key, value]) => `<p><strong>${key}:</strong> ${value.toFixed(3)}</p>`).join('')}
                            <h4>Solubility</h4>
                            ${Object.entries(variant.features.Solubility).map(([key, value]) => `<p><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}</p>`).join('')}
                            <h4>Drug-Likeness</h4>
                            ${Object.entries(variant.features.DrugLikeness).map(([key, value]) => `<p><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}</p>`).join('')}
                            <h4>Medicinal Chemistry</h4>
                            ${Object.entries(variant.features.MedicinalChem).map(([key, value]) => `<p><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}</p>`).join('')}
                            <h4>Pharmacokinetics</h4>
                            ${Object.entries(variant.features.Pharmacokinetics).map(([key, value]) => `<p><strong>${key}:</strong> ${value}</p>`).join('')}
                        </div>
                    </div>
                </div>
            `;
            propertyResults.appendChild(accordion);
        }

        if (endpoint === '/predict') {
            const admeSection = document.createElement('div');
            admeSection.innerHTML = '<h4>ADME Predictions</h4>';
            if (data.predictions) {
                for (const [key, value] of Object.entries(data.predictions)) {
                    if (key !== 'SMILES') {
                        const card = document.createElement('div');
                        card.className = 'property-card';
                        card.innerHTML = `<strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}`;
                        admeSection.appendChild(card);
                    }
                }
            }
            propertyResults.appendChild(admeSection);

            const detailedSection = document.createElement('div');
            detailedSection.innerHTML = '<h4>Detailed Molecular Analysis</h4>';
            if (data.detailed_analysis && !data.detailed_analysis.error) {
                detailedSection.innerHTML += `
                    <h5>Properties</h5>
                    ${Object.entries(data.detailed_analysis.properties).map(([key, value]) => `<p><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(3) : value}</p>`).join('')}
                    <h5>Druglikeness</h5>
                    <p><strong>Druglikeness:</strong> ${data.detailed_analysis.druglikeness}</p>
                `;
            } else {
                detailedSection.innerHTML += '<p>No detailed analysis available</p>';
            }
            propertyResults.appendChild(detailedSection);

            const rulesSection = document.createElement('div');
            rulesSection.innerHTML = '<h4>Rule-Based ADMET Predictions</h4>';
            if (data.rules_admet && typeof data.rules_admet === 'object') {
                for (const [property, details] of Object.entries(data.rules_admet)) {
                    const card = document.createElement('div');
                    card.className = `property-card ${details.prediction.toLowerCase().includes('good') || details.prediction.toLowerCase().includes('yes') || details.prediction.toLowerCase().includes('penetrant') ? 'good' : 'poor'}`;
                    card.innerHTML = `
                        <strong>${property}:</strong> ${details.prediction}<br>
                        <small>Confidence: ${(details.confidence * 100).toFixed(2)}%</small>
                    `;
                    rulesSection.appendChild(card);
                }
            } else {
                rulesSection.innerHTML += '<p>No ADMET data available</p>';
            }
            propertyResults.appendChild(rulesSection);

            renderProperties(data.original, "Original Molecule");
            renderProperties(data.optimized, "ACO-Optimized Molecule");

            document.getElementById('optimizedRadarTitle').textContent = "ACO-Optimized Radar";
            const originalRadar = document.getElementById('originalRadar');
            originalRadar.src = data.original?.radar ? `data:image/png;base64,${data.original.radar}` : '';
            originalRadar.style.display = data.original?.radar ? 'block' : 'none';

            const optimizedRadar = document.getElementById('optimizedRadar');
            optimizedRadar.src = data.optimized?.radar ? `data:image/png;base64,${data.optimized.radar}` : '';
            optimizedRadar.style.display = data.optimized?.radar ? 'block' : 'none';

            document.getElementById('improvements').innerHTML = '';
        } else if (endpoint === '/optimize_with_gemini') {
            renderProperties(data.original, "Original Molecule");
            renderProperties(data.gemini_optimized, "Gemini-Optimized Molecule");

            document.getElementById('optimizedRadarTitle').textContent = "Gemini-Optimized Radar";
            const originalRadar = document.getElementById('originalRadar');
            originalRadar.src = data.original?.radar ? `data:image/png;base64,${data.original.radar}` : '';
            originalRadar.style.display = data.original?.radar ? 'block' : 'none';

            const optimizedRadar = document.getElementById('optimizedRadar');
            optimizedRadar.src = data.gemini_optimized?.radar ? `data:image/png;base64,${data.gemini_optimized.radar}` : '';
            optimizedRadar.style.display = data.gemini_optimized?.radar ? 'block' : 'none';

            const improvements = document.getElementById('improvements');
            improvements.innerHTML = `
                <h4>Improvements from Original</h4>
                <table class="table">
                    <thead>
                        <tr>
                            <th>Property</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>QED</td>
                            <td>${data.improvements.QED.toFixed(3)} (${data.improvements.QED > 0 ? 'Improved' : 'Decreased'})</td>
                        </tr>
                        <tr>
                            <td>LogP</td>
                            <td>${data.improvements.LogP.toFixed(3)} (${data.improvements.LogP > 0 ? 'Increased' : 'Decreased'})</td>
                        </tr>
                        <tr>
                            <td>Toxicity</td>
                            <td>${data.improvements.Toxicity.toFixed(3)} (${data.improvements.Toxicity > 0 ? 'Reduced' : 'Increased'})</td>
                        </tr>
                        <tr>
                            <td>Fitness</td>
                            <td>${data.improvements.Fitness.toFixed(3)} (${data.improvements.Fitness > 0 ? 'Improved' : 'Decreased'})</td>
                        </tr>
                    </tbody>
                </table>
            `;
        }
    } catch (error) {
        console.error('Fetch Error:', error);
        alert('Failed to fetch data. Please check the server and try again.');
        document.getElementById('results').style.display = 'none';
    }
}

// Drug Discovery Functions
async function loadTarget() {
    const proteinId = document.getElementById('proteinId').value.trim();
    if (!proteinId) {
        showError('Please enter a protein ID or common name');
        return;
    }
    showLoading();
    showError('');
    try {
        const response = await fetch(`${BASE_URL}/api/target?proteinId=${encodeURIComponent(proteinId)}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Server error');
        const targetDiv = document.getElementById('targetInfo');
        targetDiv.innerHTML = `
            <h3>${data.protein_name}</h3>
            <p><strong>Accession:</strong> ${data.accession} <button class="molstar-btn" onclick="viewInMolstar('${data.accession}')">View in Mol*</button></p>
            <p><strong>Gene:</strong> ${data.gene_names}</p>
            <p><strong>Organism:</strong> ${data.organism}</p>
            <div class="function-details">
                <h4>Detailed Functions:</h4>
                <ul>
                    ${data.functions.map(func => `
                        <li>
                            ${func.description}
                            ${func.references?.length > 0 ? `
                                <div class="references">
                                    <small>References: ${func.references.map(ref => `<a href="https://pubmed.ncbi.nlm.nih.gov/${ref}" target="_blank">PMID:${ref}</a>`).join(', ')}</small>
                                </div>
                            ` : ''}
                        </li>
                    `).join('')}
                </ul>
            </div>
        `;
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

async function loadCompounds() {
    const proteinId = document.getElementById('proteinId').value.trim();
    if (!proteinId) {
        showError('Please search for a target first');
        return;
    }
    showLoading();
    showError('');
    try {
        const response = await fetch(`${BASE_URL}/api/chembl_leads?proteinId=${encodeURIComponent(proteinId)}`);
        const data = await response.json();
        if (!response.ok || !data.results) throw new Error(data.error || 'No compounds found');
        $('#compoundsTable').DataTable({
            destroy: true,
            data: data.results,
            columns: [
                { data: 'chembl_id' },
                {
                    data: 'smiles',
                    render: function(data) {
                        return `<span>${data} <button class="copy-btn" onclick="copyToClipboard('${data}')">Copy</button></span>`;
                    }
                },
                {
                    data: 'pdb_id',
                    render: function(data) {
                        return `<span>${data} <button class="copy-btn" onclick="copyToClipboard('${data}')">Copy</button></span>`;
                    }
                },
                { data: 'type' },
                { data: 'value', render: (data, type, row) => `${data} ${row.units || ''}` },
                { data: 'molecular_mass' },
                { data: 'logp' },
                { data: 'sdf_file', render: (data) => `<button onclick="view3DStructure('/uploads/${data}')">View 3D</button>` },
                { data: null, render: (data, type, row) => `<button class="optimize-btn" data-smiles="${row.smiles}" data-ic50="${row.ic50_value}">Optimize</button>` }
            ],
            order: [[3, 'asc']]
        });
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

async function optimizeLead(leadSmiles, leadIc50) {
    const table = $('#compoundsTable').DataTable();
    const allCompounds = table.rows().data().toArray().map(row => ({
        smiles: row.smiles,
        sdf_file: row.sdf_file,
        pdb_id: row.pdb_id,
        ic50_value: row.ic50_value
    })).filter(c => c.smiles !== leadSmiles && c.ic50_value !== null);

    showLoading();
    showError('');
    try {
        const response = await fetch('/api/optimize', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ leadSmiles, leadIc50, allCompounds })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Optimization failed');
        const resultsDiv = document.getElementById('optimizationResults');
        resultsDiv.innerHTML = `
            <h3>Lead Optimization for ${leadSmiles}</h3>
            <p><strong>Lead Compound Properties:</strong></p>
            <ul>
                <li>Molecular Mass: ${data.properties.molecular_mass} Da</li>
                <li>LogP: ${data.properties.logP}</li>
                <li>Solubility (logS): ${data.properties.solubility}</li>
                <li>H-bond Donors: ${data.properties.hbd}</li>
                <li>H-bond Acceptors: ${data.properties.hba}</li>
                <li>PSA: ${data.properties.psa} Å²</li>
                <li>Rotatable Bonds: ${data.properties.rotatable_bonds}</li>
                <li>PDB ID: ${data.properties.pdb_id} <button class="copy-btn" onclick="copyToClipboard('${data.properties.pdb_id}')">Copy</button></li>
            </ul>
            <p><strong>Lipinski’s Rule Violations:</strong></p>
            <ul>${data.lipinski_violations.length > 0 ? data.lipinski_violations.map(v => `<li>${v}</li>`).join('') : '<li>None</li>'}</ul>
            <p><strong>Optimization Suggestions:</strong></p>
            <ul>${data.suggestions.map(s => `<li>${s}</li>`).join('')}</ul>
            <p><strong>Similar Compounds with Better IC50:</strong></p>
            <table>
                <thead>
                    <tr>
                        <th>SMILES</th>
                        <th>PDB ID</th>
                        <th>IC50 (nM)</th>
                        <th>Similarity</th>
                        <th>Molecular Mass</th>
                        <th>LogP</th>
                        <th>Solubility</th>
                        <th>HBD</th>
                        <th>HBA</th>
                        <th>PSA</th>
                        <th>3D View</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.similarCompounds.map(c => `
                        <tr>
                            <td>${c.smiles} <button class="copy-btn" onclick="copyToClipboard('${c.smiles}')">Copy</button></td>
                            <td>${c.pdb_id} <button class="copy-btn" onclick="copyToClipboard('${c.pdb_id}')">Copy</button></td>
                            <td>${c.ic50_value}</td>
                            <td>${c.similarity.toFixed(2)}</td>
                            <td>${c.molecular_mass}</td>
                            <td>${c.logp}</td>
                            <td>${c.solubility}</td>
                            <td>${c.hbd}</td>
                            <td>${c.hba}</td>
                            <td>${c.psa}</td>
                            <td><button onclick="view3DStructure('/uploads/${c.sdf_file}')">View</button></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        view3DStructure('/uploads/' + data.properties.sdf_file);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// NGL Viewer Functions
function view3DStructure(sdfUrl) {
    var stage = new NGL.Stage('viewer', { backgroundColor: 'white' });
    stage.removeAllComponents(); // Clear previous content
    stage.loadFile(sdfUrl, { ext: 'sdf' }).then(function(component) {
        component.addRepresentation('ball+stick');
        component.autoView();
    }).catch(function(err) {
        console.error('Error loading SDF:', err);
    });
}

// Call it with the correct URL


function view3DBySmiles() {
    const smiles = document.getElementById('smilesInput3D').value.trim();
    if (!smiles) {
        showError('Please enter a SMILES string');
        return;
    }
    if (!stage3D) stage3D = new NGL.Stage('viewer3D', { backgroundColor: 'white' });
    stage3D.removeAllComponents();
    fetch('/api/optimize', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ leadSmiles: smiles, leadIc50: 0, allCompounds: [] })
    })
    .then(response => {
        if (!response.ok) throw new Error('Failed to fetch SDF');
        return response.json();
    })
    .then(data => {
        const sdfUrl = '/uploads/' + data.properties.sdf_file;
        console.log('Loading SDF from:', sdfUrl);
        stage3D.loadFile(sdfUrl, { ext: 'sdf' }).then(function(component) {
            component.addRepresentation('ball+stick');
            component.autoView();
            if (spinning) stage3D.animationControls.play({ direction: 'spin' });
        }).catch(err => showError('Failed to load SDF: ' + err.message));
    })
    .catch(err => showError('Failed to fetch SDF: ' + err.message));
}

function setNGLStyle(style) {
    [stage, stage3D].forEach(s => {
        if (!s) return;
        s.eachComponent(function(component) {
            component.removeAllRepresentations();
            component.addRepresentation(style);
            component.autoView();
        });
    });
}

function addSurface(type) {
    const surfaceParams = {
        'vdw': { surfaceType: 'vdw', opacity: 0.7, color: 'grey' },
        'sas': { surfaceType: 'sas', opacity: 0.7, color: 'blue' },
        'ses': { surfaceType: 'ses', opacity: 0.7, color: 'green' }
    }[type] || {};
    [stage, stage3D].forEach(s => {
        if (!s) return;
        s.eachComponent(function(component) {
            component.addRepresentation('surface', surfaceParams);
            component.autoView();
        });
    });
}

function removeSurface() {
    [stage, stage3D].forEach(s => {
        if (!s) return;
        s.eachComponent(function(component) {
            component.removeRepresentation('surface');
            component.autoView();
        });
    });
}

function measureNGLDistance() {
    const s = stage || stage3D;
    if (!s) return;
    alert('Click two atoms to measure distance. Result will be logged to console.');
    s.signals.clicked.add(function(pickingProxy) {
        if (pickingProxy && pickingProxy.atom) {
            const selected = s.compList[0].selection.atoms || [];
            if (selected.length === 2) {
                const pos1 = selected[0].globalposition;
                const pos2 = selected[1].globalposition;
                const distance = Math.sqrt(
                    Math.pow(pos2.x - pos1.x, 2) +
                    Math.pow(pos2.y - pos1.y, 2) +
                    Math.pow(pos2.z - pos1.z, 2)
                );
                console.log('Distance:', distance.toFixed(2), 'Å');
                s.compList[0].selection.clear();
            }
        }
    });
}

function measureNGLAngle() {
    const s = stage || stage3D;
    if (!s) return;
    alert('Click three atoms to measure angle. Result will be logged to console.');
    s.signals.clicked.add(function(pickingProxy) {
        if (pickingProxy && pickingProxy.atom) {
            const selected = s.compList[0].selection.atoms || [];
            if (selected.length === 3) {
                const pos1 = selected[0].globalposition;
                const pos2 = selected[1].globalposition;
                const pos3 = selected[2].globalposition;
                const v1 = { x: pos1.x - pos2.x, y: pos1.y - pos2.y, z: pos1.z - pos2.z };
                const v2 = { x: pos3.x - pos2.x, y: pos3.y - pos2.y, z: pos3.z - pos2.z };
                const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
                const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
                const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
                const angle = Math.acos(dot / (mag1 * mag2)) * (180 / Math.PI);
                console.log('Angle:', angle.toFixed(2), 'degrees');
                s.compList[0].selection.clear();
            }
        }
    });
}

function measureNGLDihedral() {
    const s = stage || stage3D;
    if (!s) return;
    alert('Click four atoms to measure dihedral angle. Result will be logged to console.');
    s.signals.clicked.add(function(pickingProxy) {
        if (pickingProxy && pickingProxy.atom) {
            const selected = s.compList[0].selection.atoms || [];
            if (selected.length === 4) {
                const pos1 = selected[0].globalposition;
                const pos2 = selected[1].globalposition;
                const pos3 = selected[2].globalposition;
                const pos4 = selected[3].globalposition;
                const v1 = { x: pos2.x - pos1.x, y: pos2.y - pos1.y, z: pos2.z - pos1.z };
                const v2 = { x: pos3.x - pos2.x, y: pos3.y - pos2.y, z: pos3.z - pos2.z };
                const v3 = { x: pos4.x - pos3.x, y: pos4.y - pos3.y, z: pos4.z - pos3.z };
                const n1 = {
                    x: v1.y * v2.z - v1.z * v2.y,
                    y: v1.z * v2.x - v1.x * v2.z,
                    z: v1.x * v2.y - v1.y * v2.x
                };
                const n2 = {
                    x: v2.y * v3.z - v2.z * v3.y,
                    y: v2.z * v3.x - v2.x * v3.z,
                    z: v2.x * v3.y - v2.y * v3.x
                };
                const dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
                const mag1 = Math.sqrt(n1.x * n1.x + n1.y * n1.y + n1.z * n1.z);
                const mag2 = Math.sqrt(n2.x * n2.x + n2.y * n2.y + n2.z * n2.z);
                const dihedral = Math.acos(dot / (mag1 * mag2)) * (180 / Math.PI);
                console.log('Dihedral Angle:', dihedral.toFixed(2), 'degrees');
                s.compList[0].selection.clear();
            }
        }
    });
}

function toggleLabels() {
    labelsVisible = !labelsVisible;
    [stage, stage3D].forEach(s => {
        if (!s) return;
        s.eachComponent(function(component) {
            if (labelsVisible) {
                component.addRepresentation('label', { labelType: 'atomname', color: 'white', radius: 0.5 });
            } else {
                component.removeRepresentation('label');
            }
            component.autoView();
        });
    });
}

function zoomIn() {
    [stage, stage3D].forEach(s => {
        if (s) s.viewer.zoom(-0.5);
    });
}

function zoomOut() {
    [stage, stage3D].forEach(s => {
        if (s) s.viewer.zoom(0.5);
    });
}

function toggleSpin() {
    spinning = !spinning;
    [stage, stage3D].forEach(s => {
        if (!s) return;
        if (spinning) {
            s.animationControls.play({ direction: 'spin' });
        } else {
            s.animationControls.pause();
        }
    });
}

function setColorScheme(scheme) {
    [stage, stage3D].forEach(s => {
        if (!s) return;
        s.eachComponent(function(component) {
            component.eachRepresentation(function(rep) {
                rep.setParameters({ colorScheme: scheme });
            });
            component.autoView();
        });
    });
}

function toggleBackground() {
    [stage, stage3D].forEach(s => {
        if (!s) return;
        const currentColor = s.backgroundColor === 'white' ? 'black' : 'white';
        s.setParameters({ backgroundColor: currentColor });
    });
}
