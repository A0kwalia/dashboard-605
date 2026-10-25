const KSIModel = require(require("path").resolve(process.argv[2]));
const fs = require("fs");
const m = JSON.parse(fs.readFileSync(process.argv[3]));
const fx = JSON.parse(fs.readFileSync(process.argv[4]));
const model = new KSIModel(m);
let maxd = 0, t = Date.now();
fx.rows.forEach((r, i) => { const d = Math.abs(model.predict(r) - fx.p[i]); if (d > maxd) maxd = d; });
console.log("rows", fx.rows.length, "max abs diff", maxd.toExponential(3), "ms", Date.now() - t);
// explanation timing on one row
const groups = m.categorical.filter(c => !["LGA","Conurbation 1"].includes(c)).map(c => ({label:c, features:[c]}));
groups.push({label:"loc", features:["LGA","Latitude","Longitude","Conurbation 1"]}, {label:"speed", features:["SPEED_LIMIT_KMH"]});
t = Date.now(); const ex = model.explain(fx.rows[0], groups);
console.log("explain ms", Date.now() - t, "p", ex.p.toFixed(4), ex.factors.slice(0,5).map(f => f.group.label + " " + (f.delta*100).toFixed(2)).join(" | "));
