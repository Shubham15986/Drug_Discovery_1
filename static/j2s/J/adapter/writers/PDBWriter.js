Clazz.declarePackage("J.adapter.writers");
Clazz.load(["J.api.JmolWriter"], "J.adapter.writers.PDBWriter", ["java.util.Arrays", "$.Date", "$.Hashtable", "JU.Lst", "$.P3", "$.PT", "JM.LabelToken", "JV.Viewer"], function(){
var c$ = Clazz.decorateAsClass(function(){
this.vwr = null;
this.oc = null;
this.isPQR = false;
this.doTransform = false;
this.allTrajectories = false;
Clazz.instantialize(this, arguments);}, J.adapter.writers, "PDBWriter", null, J.api.JmolWriter);
/*LV!1824 unnec constructor*/Clazz.overrideMethod(c$, "set", 
function(viewer, oc, data){
this.vwr = viewer;
this.oc = (oc == null ? this.vwr.getOutputChannel(null, null) : oc);
this.isPQR = (data[0]).booleanValue();
this.doTransform = (data[1]).booleanValue();
this.allTrajectories = (data[2]).booleanValue();
}, "JV.Viewer,JU.OC,~A");
Clazz.overrideMethod(c$, "write", 
function(bs){
var type = this.oc.getType();
this.isPQR = new Boolean (this.isPQR | (type != null && type.indexOf("PQR") >= 0)).valueOf();
this.doTransform = new Boolean (this.doTransform | (type != null && type.indexOf("-coord true") >= 0)).valueOf();
var atoms = this.vwr.ms.at;
var models = this.vwr.ms.am;
var occTemp = "%6.2Q%6.2b          ";
if (this.isPQR) {
occTemp = "%8.4P%7.4V       ";
var charge = 0;
for (var i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) charge += atoms[i].getPartialCharge();

this.oc.append("REMARK   1 PQR file generated by Jmol " + JV.Viewer.getJmolVersion()).append("\nREMARK   1 " + "created " + ( new java.util.Date())).append("\nREMARK   1 Forcefield Used: unknown\nREMARK   1").append("\nREMARK   5").append("\nREMARK   6 Total charge on this protein: " + charge + " e\nREMARK   6\n");
}var iModel = atoms[bs.nextSetBit(0)].mi;
var iModelLast = -1;
var lastAtomIndex = bs.length() - 1;
var lastModelIndex = atoms[lastAtomIndex].mi;
var isMultipleModels = (iModel != lastModelIndex);
var bsModels = this.vwr.ms.getModelBS(bs, true);
var nModels = bsModels.cardinality();
var lines =  new JU.Lst();
var isMultipleBondPDB = models[iModel].isPdbWithMultipleBonds;
var uniqueAtomNumbers = false;
if (nModels > 1) {
var conectArray = null;
for (var nFiles = 0, i = bsModels.nextSetBit(0); i >= 0; i = bsModels.nextSetBit(i + 1)) {
var a = this.vwr.ms.getModelAuxiliaryInfo(i).get("PDB_CONECT_bonds");
if (a !== conectArray || !this.vwr.ms.am[i].isBioModel) {
conectArray = a;
if (nFiles++ > 0) {
uniqueAtomNumbers = true;
break;
}}}
}var tokens;
var ptTemp =  new JU.P3();
var o =  Clazz.newArray(-1, [ptTemp]);
var q = (this.doTransform ? this.vwr.tm.getRotationQ() : null);
var map =  new java.util.Hashtable();
var isBiomodel = false;
var firstAtomIndexNew = (uniqueAtomNumbers ?  Clazz.newIntArray (/*org.eclipse.jdt.core.dom.SimpleName*/nModels, 0) : null);
var modelPt = 0;
for (var i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) {
var a = atoms[i];
isBiomodel = models[a.mi].isBioModel;
if (isMultipleModels && a.mi != iModelLast) {
if (iModelLast != -1) {
modelPt = this.fixPDBFormat(lines, map, this.oc, firstAtomIndexNew, modelPt);
this.oc.append("ENDMDL\n");
}lines =  new JU.Lst();
iModel = iModelLast = a.mi;
this.oc.append("MODEL     " + (iModelLast + 1) + "\n");
}var sa = a.getAtomName();
var leftJustify = (a.getElementSymbol().length == 2 || sa.length >= 4 || JU.PT.isDigit(sa.charAt(0)));
var isHetero = a.isHetero();
if (!isBiomodel) tokens = (leftJustify ? JM.LabelToken.compile(this.vwr, "HETATM%5.-5i %-4.4a%1AUNK %1c   1%1E   _XYZ_" + occTemp, '\0', null) : JM.LabelToken.compile(this.vwr, "HETATM%5.-5i  %-3.3a%1AUNK %1c   1%1E   _XYZ_" + occTemp, '\0', null));
 else if (isHetero) tokens = (leftJustify ? JM.LabelToken.compile(this.vwr, "HETATM%5.-5i %-4.4a%1A%3.3n %1c%4.-4R%1E   _XYZ_" + occTemp, '\0', null) : JM.LabelToken.compile(this.vwr, "HETATM%5.-5i  %-3.3a%1A%3.3n %1c%4.-4R%1E   _XYZ_" + occTemp, '\0', null));
 else tokens = (leftJustify ? JM.LabelToken.compile(this.vwr, "ATOM  %5.-5i %-4.4a%1A%3.3n %1c%4.-4R%1E   _XYZ_" + occTemp, '\0', null) : JM.LabelToken.compile(this.vwr, "ATOM  %5.-5i  %-3.3a%1A%3.3n %1c%4.-4R%1E   _XYZ_" + occTemp, '\0', null));
var XX = a.getElementSymbolIso(false).toUpperCase();
XX = this.pdbKey(a.group.getBioPolymerIndexInModel()) + this.pdbKey(a.group.groupIndex) + JM.LabelToken.formatLabelAtomArray(this.vwr, a, tokens, '\0', null, ptTemp) + (XX.length == 1 ? " " + XX : XX.substring(0, 2)) + "  ";
this.vwr.ms.getPointTransf(-1, a, q, ptTemp);
var xyz = JU.PT.sprintf("%8.3p%8.3p%8.3p", "p", o);
if (xyz.length > 24) xyz = JU.PT.sprintf("%8.2p%8.2p%8.2p", "p", o);
XX = JU.PT.rep(XX, "_XYZ_", xyz);
lines.addLast(XX);
}
this.fixPDBFormat(lines, map, this.oc, firstAtomIndexNew, modelPt);
if (isMultipleModels) this.oc.append("ENDMDL\n");
modelPt = -1;
iModelLast = -1;
var conectKey = "" + (isMultipleModels ? modelPt : 0);
isBiomodel = false;
for (var i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i + 1)) {
var a = atoms[i];
if (a.mi != iModelLast) {
var m = models[a.mi];
iModelLast = a.mi;
isBiomodel = m.isBioModel;
modelPt++;
}var isHetero = (!isBiomodel || a.isHetero());
var isCysS = !isHetero && (a.getElementNumber() == 16);
if (isHetero || isMultipleBondPDB || isCysS) {
var bonds = a.bonds;
if (bonds == null) continue;
for (var j = 0; j < bonds.length; j++) {
var b = bonds[j];
var iThis = a.getAtomNumber();
var a2 = b.getOtherAtom(a);
if (!bs.get(a2.i)) continue;
var n = b.getCovalentOrder();
if (n == 1 && (isMultipleBondPDB && !isHetero && !isCysS || isCysS && a2.getElementNumber() != 16)) continue;
var iOther = a2.getAtomNumber();
switch (n) {
case 2:
case 3:
if (iOther < iThis) continue;
case 1:
var inew = map.get(conectKey + "." + Integer.$valueOf(iThis));
var inew2 = map.get(conectKey + "." + Integer.$valueOf(iOther));
if (inew == null || inew2 == null) break;
this.oc.append("CONECT").append(JU.PT.formatStringS("%5s", "s", "" + inew));
var s = JU.PT.formatStringS("%5s", "s", "" + inew2);
for (var k = 0; k < n; k++) this.oc.append(s);

this.oc.append("\n");
break;
}
}
}}
return this.toString();
}, "JU.BS");
Clazz.defineMethod(c$, "pdbKey", 
function(np){
var xp = (np < 0 ? "~999" : "   " + np);
return xp.substring(xp.length - 4);
}, "~N");
Clazz.defineMethod(c$, "fixPDBFormat", 
function(lines, map, out, firstAtomIndexNew, modelPt){
lines.addLast("~999~999XXXXXX99999999999999999999~99~");
var alines =  new Array(lines.size());
lines.toArray(alines);
java.util.Arrays.sort(alines);
lines.clear();
for (var i = 0, n = alines.length; i < n; i++) {
lines.addLast(alines[i]);
}
var lastPoly = null;
var lastLine = null;
var n = lines.size();
var newAtomNumber = 0;
var iBase = (firstAtomIndexNew == null ? 0 : firstAtomIndexNew[modelPt]);
for (var i = 0; i < n; i++) {
var s = lines.get(i);
var poly = s.substring(0, 4);
s = s.substring(8);
var isTerm = false;
var isLast = (s.indexOf("~99~") >= 0);
if (!poly.equals(lastPoly) || isLast) {
if (lastPoly != null && !lastPoly.equals("~999")) {
isTerm = true;
s = "TER   " + lastLine.substring(6, 11) + "      " + lastLine.substring(17, 27);
lines.add(i, poly + "~~~~" + s);
n++;
}lastPoly = poly;
}if (isLast && !isTerm) break;
lastLine = s;
newAtomNumber = i + 1 + iBase;
if (map != null && !isTerm) map.put("" + modelPt + "." + Integer.$valueOf(JU.PT.parseInt(s.substring(6, 11))), Integer.$valueOf(newAtomNumber));
var si = "     " + newAtomNumber;
out.append(s.substring(0, 6)).append(si.substring(si.length - 5)).append(s.substring(11)).append("\n");
}
if (firstAtomIndexNew != null && ++modelPt < firstAtomIndexNew.length) firstAtomIndexNew[modelPt] = newAtomNumber;
return modelPt;
}, "JU.Lst,java.util.Map,JU.OC,~A,~N");
Clazz.overrideMethod(c$, "toString", 
function(){
return (this.oc == null ? "" : this.oc.toString());
});
});
;//5.0.1-v4 Thu Dec 12 22:51:35 CST 2024
