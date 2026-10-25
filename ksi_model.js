/* KSI risk model, evaluated in the browser.
 *
 * Loads model_ksi.json (exported from the trained, isotonic-calibrated gradient boosting model)
 * and reproduces its predictions without a server. A row is an array aligned to
 * model.features: categorical features as vocabulary codes, numeric features as numbers
 * (null = not recorded, replaced by the training median as in the Python pipeline).
 */
(function (root) {
  "use strict";

  function KSIModel(m) {
    this.m = m;
    this.F = m.features;
    this.fi = {};
    for (var i = 0; i < this.F.length; i++) this.fi[this.F[i]] = i;
    this.numIdx = {};
    for (var k = 0; k < m.numeric.length; k++) this.numIdx[this.fi[m.numeric[k]]] = k;
    this.code = {};
    for (var c in m.vocab) {
      var d = {};
      m.vocab[c].forEach(function (v, j) { d[v] = j; });
      this.code[c] = d;
    }
    // categorical split sets -> byte masks over the feature's vocabulary
    var self = this;
    m.folds.forEach(function (fold) {
      fold.masks = fold.sets.map(function () { return null; });
      fold.trees.forEach(function (tree) {
        tree.forEach(function (nd) {
          if (nd[1] === "s" && !fold.masks[nd[2]]) {
            var size = m.vocab[self.F[nd[0]]].length, mask = new Uint8Array(size);
            fold.sets[nd[2]].forEach(function (c) { mask[c] = 1; });
            fold.masks[nd[2]] = mask;
          }
        });
      });
    });
  }

  KSIModel.prototype._raw = function (fold, row) {
    var s = fold.base, trees = fold.trees, med = fold.median, numIdx = this.numIdx;
    for (var t = 0; t < trees.length; t++) {
      var tree = trees[t], nd = tree[0];
      while (nd[0] !== -1) {
        var right;
        if (nd[1] === "s") {
          right = fold.masks[nd[2]][row[nd[0]]] === 1;
        } else {
          var x = row[nd[0]];
          if (x === null || x === undefined || x !== x) x = med[numIdx[nd[0]]];
          right = !(x <= nd[2]);
        }
        nd = tree[right ? nd[4] : nd[3]];
      }
      s += nd[1];
    }
    return s;
  };

  /* Calibrated probability of a KSI outcome for one row: each fold's gradient-boosting score
   * goes through that fold's sigmoid (Platt) calibrator, and the three are averaged. */
  KSIModel.prototype.predict = function (row) {
    var folds = this.m.folds, p = 0;
    for (var f = 0; f < folds.length; f++) {
      var fd = folds[f];
      p += 1 / (1 + Math.exp(fd.sig_a * this._raw(fd, row) + fd.sig_b));
    }
    return p / folds.length;
  };

  /* Build a row from a {feature: value} object; categorical values are vocabulary strings. */
  KSIModel.prototype.row = function (obj) {
    var self = this;
    return this.F.map(function (f) {
      var v = obj[f];
      if (self.code[f]) return v in self.code[f] ? self.code[f][v] : 0;
      return v === null || v === undefined || v === "" ? null : +v;
    });
  };

  /* Contribution of each factor group to this prediction.
   * For a group, contribution = p(row) - mean over background crashes of p(row with that group's
   * values taken from the background crash). Positive = this value raises the risk compared with
   * the conditions of a typical reported crash; negative = lowers it. */
  KSIModel.prototype.explain = function (row, groups) {
    var self = this, bg = this.m.background, base = this.predict(row);
    var out = groups.map(function (g) {
      var idx = g.features.map(function (f) { return self.fi[f]; }), sum = 0;
      for (var b = 0; b < bg.length; b++) {
        var r = row.slice();
        for (var k = 0; k < idx.length; k++) r[idx[k]] = bg[b][idx[k]];
        sum += self.predict(r);
      }
      return { group: g, delta: base - sum / bg.length };
    });
    out.sort(function (a, b) { return Math.abs(b.delta) - Math.abs(a.delta); });
    return { p: base, factors: out };
  };

  root.KSIModel = KSIModel;
  if (typeof module !== "undefined") module.exports = KSIModel;
})(typeof window !== "undefined" ? window : globalThis);
