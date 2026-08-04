/* The page.
 *
 * One tree, drawn from the arcs, with the code lineage over the top. Everything
 * it shows is in the JSON above it — nothing is fetched, and nothing is worked
 * out here that the model could have worked out once.
 *
 * A card rather than a coloured blob: the score band is a chip on the left and
 * the score itself is printed beside it, so colour is a way to see the shape of
 * the run at a glance and never the only way to read a value. The text stays in
 * ink for the same reason.
 */
(function () {
  "use strict";

  var data = JSON.parse(document.getElementById("run-data").textContent);

  var CARD_W = 340;
  var CARD_H = 46;
  // Siblings side by side and depth downward: the root is at the top and an
  // idea's children hang under it, which is the way the search itself is
  // described. The gap between siblings is the card plus air; the gap between
  // depths is the card plus room for the arc to be seen.
  var SIBLING_GAP = 380;
  var DEPTH_GAP = 132;
  // How far a lineage curve sits to the right of the tree arc it parallels.
  var LINEAGE_OFFSET = 22;
  var LABEL_CHARS = 44;
  var ID_CHARS = 10;
  var RAMP_STEPS = 5;

  var fmt = d3.format(".6~g");

  var svg = d3.select("#tree");
  var panel = document.getElementById("panel");
  var filterInput = document.getElementById("filter");
  var lineageSelect = document.getElementById("lineage-mode");

  // --- the hierarchy --------------------------------------------------------

  var present = new Set(data.nodes.map(function (n) { return n.id; }));
  var orphans = data.nodes.filter(function (n) {
    return n.parent_node_id !== null && !present.has(n.parent_node_id);
  });

  if (orphans.length) {
    // An arc naming a parent that is not in the file. Said out loud rather than
    // quietly dropped: a tree with a piece missing is worse than no tree.
    svg
      .append("text")
      .attr("class", "label")
      .attr("x", 24)
      .attr("y", 40)
      .text(
        orphans.length + " node(s) hang off a parent that is not in arcs.json: " +
          orphans.map(function (n) { return n.id; }).join(", ")
      );
    return;
  }

  var root = d3
    .stratify()
    .id(function (d) { return d.id; })
    .parentId(function (d) { return d.parent_node_id; })(data.nodes);

  // Siblings in the order the search made them, so reading down a fan is
  // reading forward in time.
  root.sort(function (a, b) {
    var ai = a.data.created_at_iteration;
    var bi = b.data.created_at_iteration;

    return (
      (ai === null ? Infinity : ai) - (bi === null ? Infinity : bi) ||
      d3.ascending(a.id, b.id)
    );
  });

  d3
    .tree()
    .nodeSize([SIBLING_GAP, DEPTH_GAP])
    .separation(function (a, b) { return a.parent === b.parent ? 1 : 1.15; })(root);

  var byId = new Map(root.descendants().map(function (d) { return [d.id, d]; }));

  // The root-to-best spine. One line in this tree is worth following and this is
  // it; every other arc stays hairline.
  var winning = new Set();

  if (data.run.best_node_id && byId.has(data.run.best_node_id)) {
    byId.get(data.run.best_node_id).ancestors().forEach(function (a) {
      winning.add(a.id);
    });
  }

  // The model hands back fewer bands than the ramp has steps when the bests tie
  // or there are only a few of them. Spread what there is over the whole ramp
  // rather than crowding it into the dark end.
  function rampStep(band) {
    if (data.run.bands <= 1) return 0;

    return Math.round((band * (RAMP_STEPS - 1)) / (data.run.bands - 1));
  }

  // --- drawing --------------------------------------------------------------

  var defs = svg.append("defs");

  defs
    .append("marker")
    .attr("id", "lineage-arrow")
    .attr("viewBox", "0 0 8 8")
    .attr("refX", 7)
    .attr("refY", 4)
    .attr("markerWidth", 6)
    .attr("markerHeight", 6)
    .attr("orient", "auto-start-reverse")
    .append("path")
    .attr("class", "arrowhead")
    .attr("d", "M0,0 L8,4 L0,8 z");

  // One clip path for every card: each node's group has its own origin, so the
  // same rectangle cuts each of them at its own edge. A constraint nobody
  // budgeted for overflows into a neighbour otherwise.
  defs
    .append("clipPath")
    .attr("id", "card-clip")
    .append("rect")
    .attr("width", CARD_W)
    .attr("height", CARD_H)
    .attr("rx", 6);

  var viewport = svg.append("g");
  var gLinks = viewport.append("g");
  var gLineage = viewport.append("g");
  var gNodes = viewport.append("g");

  gLinks
    .selectAll("path")
    .data(root.links())
    .join("path")
    .attr("class", function (d) {
      return winning.has(d.source.id) && winning.has(d.target.id)
        ? "link winning"
        : "link";
    })
    .attr("d", treePath);

  // A card sits centred on its layout position, so `d.x` is its middle and
  // `d.y` its top. Arcs run from the bottom of the parent to the top of the
  // child.
  function treePath(d) {
    var x1 = d.source.x;
    var y1 = d.source.y + CARD_H;
    var x2 = d.target.x;
    var y2 = d.target.y;
    var my = (y1 + y2) / 2;

    return (
      "M" + x1 + "," + y1 +
      "C" + x1 + "," + my + " " + x2 + "," + my + " " + x2 + "," + y2
    );
  }

  // Lineage collapses to one edge per pair of nodes: the same borrowing done
  // three times is one relationship, and the attempts behind it are in the panel.
  var lineage = collapse(data.lineage);

  function collapse(edges) {
    var byPair = new Map();

    edges.forEach(function (edge) {
      if (!byId.has(edge.from_node) || !byId.has(edge.to_node)) return;

      var key = edge.from_node + ">" + edge.to_node;
      var entry = byPair.get(key);

      if (!entry) {
        entry = {
          from_node: edge.from_node,
          to_node: edge.to_node,
          cross_branch: edge.cross_branch,
          edges: [],
        };
        byPair.set(key, entry);
      }

      entry.edges.push(edge);
    });

    return Array.from(byPair.values());
  }

  var lineagePaths = gLineage
    .selectAll("path")
    .data(lineage)
    .join("path")
    .attr("class", "lineage")
    .attr("marker-end", "url(#lineage-arrow)")
    .attr("d", lineagePath);

  lineagePaths.append("title").text(function (e) {
    return (
      e.edges.length + (e.edges.length === 1 ? " solution" : " solutions") +
      " started from code at " + e.from_node +
      (e.cross_branch ? " — a different branch" : "")
    );
  });

  // The same curve the tree draws, shifted sideways: a lineage edge usually runs
  // along an arc that is already there, and the useful thing to see is that it
  // does — or, for the ones that matter, that it does not. A bow scaled to the
  // distance instead swung right across the diagram and made two edges that
  // agree with the tree look like a disagreement.
  function lineagePath(e) {
    var a = byId.get(e.from_node);
    var b = byId.get(e.to_node);
    var x1 = a.x + LINEAGE_OFFSET;
    var y1 = a.y + CARD_H / 2;

    if (a === b) {
      // The code came from an earlier attempt at this very idea. A small loop
      // off the right edge, because an edge to nowhere is not readable.
      var edge = a.x + CARD_W / 2;

      return (
        "M" + edge + "," + (y1 - 11) +
        "C" + (edge + 48) + "," + (y1 - 20) + " " + (edge + 48) + "," + (y1 + 20) +
        " " + edge + "," + (y1 + 11)
      );
    }

    var x2 = b.x + LINEAGE_OFFSET;
    var y2 = b.y + CARD_H / 2;
    var my = (y1 + y2) / 2;

    return (
      "M" + x1 + "," + y1 +
      "C" + x1 + "," + my + " " + x2 + "," + my + " " + x2 + "," + y2
    );
  }

  var node = gNodes
    .selectAll("g")
    .data(root.descendants(), function (d) { return d.id; })
    .join("g")
    .attr("class", nodeClass)
    .attr("transform", function (d) {
      return "translate(" + (d.x - CARD_W / 2) + "," + d.y + ")";
    })
    .on("click", function (event, d) { open(d.id); });

  node.append("title").text(function (d) {
    return d.data.constraint === null
      ? "the root — it constrains nothing"
      : d.data.constraint;
  });

  node
    .append("rect")
    .attr("class", "card")
    .attr("width", CARD_W)
    .attr("height", CARD_H)
    .attr("rx", 6);

  var body = node.append("g").attr("clip-path", "url(#card-clip)");

  body
    .append("rect")
    .attr("class", chipClass)
    .attr("x", 10)
    .attr("y", 11)
    .attr("width", 8)
    .attr("height", CARD_H - 22)
    .attr("rx", 4);

  body
    .append("text")
    .attr("class", "label")
    .attr("x", 28)
    .attr("y", 21)
    .text(function (d) {
      return d.data.constraint === null
        ? "(the root — no constraints)"
        : clip(d.data.constraint, LABEL_CHARS);
    });

  body
    .append("text")
    .attr("class", "meta")
    .attr("x", 28)
    .attr("y", 36)
    .text(metaLine);

  node
    .filter(function (d) { return d.id === data.run.best_node_id; })
    .append("text")
    .attr("class", "badge")
    .attr("x", 0)
    .attr("y", -8)
    .text("RUN BEST");

  function nodeClass(d) {
    var classes = ["node"];

    if (!d.data.attempts) classes.push("unattempted");
    if (d.data.dead) classes.push("dead");
    if (d.id === data.run.best_node_id) classes.push("best");

    return classes.join(" ");
  }

  function chipClass(d) {
    return d.data.band === null
      ? "chip unscored"
      : "chip b" + rampStep(d.data.band);
  }

  function metaLine(d) {
    var n = d.data;
    var parts = [clip(n.id, ID_CHARS)];

    // The root holds the seed, which no iteration produced, so it has an
    // attempt and no iteration to name. Only a node nothing was ever built for
    // is "never attempted", and the two must not read the same.
    if (!n.attempts) parts.push("never attempted");
    else if (n.created_at_iteration === null) parts.push("the seed");
    else parts.push("i" + n.created_at_iteration);

    if (n.best !== null) parts.push("best " + fmt(n.best));
    else if (n.attempts) parts.push("no score");

    if (n.attempts) {
      parts.push(n.attempts + (n.attempts === 1 ? " attempt" : " attempts"));
    }

    if (n.dead) parts.push("dead");

    return parts.join(" · ");
  }

  function clip(text, width) {
    var collapsed = String(text === null || text === undefined ? "" : text)
      .split(/\s+/)
      .join(" ")
      .trim();

    return collapsed.length <= width
      ? collapsed
      : collapsed.slice(0, width - 1) + "…";
  }

  // --- zoom -----------------------------------------------------------------

  var zoom = d3
    .zoom()
    .scaleExtent([0.05, 2.5])
    .on("zoom", function (event) {
      viewport.attr("transform", event.transform);
    });

  svg.call(zoom);

  function fit(animate) {
    var nodes = root.descendants();
    var box = svg.node().getBoundingClientRect();
    var pad = 48;

    var minX = d3.min(nodes, function (d) { return d.x - CARD_W / 2; });
    // The lineage bows out to the right, so that side needs room the layout
    // does not know about.
    var maxX = d3.max(nodes, function (d) { return d.x + CARD_W / 2 + 60; });
    // The run-best badge sits above its card.
    var minY = d3.min(nodes, function (d) { return d.y - 20; });
    var maxY = d3.max(nodes, function (d) { return d.y + CARD_H; });

    var scale = Math.min(
      2.5,
      (box.width - pad * 2) / Math.max(1, maxX - minX),
      (box.height - pad * 2) / Math.max(1, maxY - minY)
    );

    var target = d3.zoomIdentity
      .translate(
        box.width / 2 - (scale * (minX + maxX)) / 2,
        box.height / 2 - (scale * (minY + maxY)) / 2
      )
      .scale(scale);

    // The first fit is not animated. A transition on load is a quarter second
    // of the page looking broken, and in any renderer that does not run
    // animation frames it never arrives at all.
    if (animate === false) svg.call(zoom.transform, target);
    else svg.transition().duration(250).call(zoom.transform, target);
  }

  d3.selectAll("[data-zoom]").on("click", function () {
    var what = this.getAttribute("data-zoom");

    if (what === "fit") fit();
    else {
      svg.transition().duration(180).call(zoom.scaleBy, what === "in" ? 1.35 : 1 / 1.35);
    }
  });

  // --- the lineage toggle ---------------------------------------------------

  function drawLineage() {
    var mode = lineageSelect.value;

    lineagePaths.attr("display", function (e) {
      if (mode === "none") return "none";
      if (mode === "all") return null;

      return e.cross_branch ? null : "none";
    });
  }

  lineageSelect.addEventListener("change", drawLineage);
  drawLineage();

  // --- the filter -----------------------------------------------------------

  function haystack(d) {
    return (d.data.id + " " + (d.data.constraint || "")).toLowerCase();
  }

  filterInput.addEventListener("input", function () {
    var query = filterInput.value.trim().toLowerCase();

    node.classed("matched", function (d) {
      return query !== "" && haystack(d).indexOf(query) !== -1;
    });
    node.classed("dimmed", function (d) {
      return query !== "" && haystack(d).indexOf(query) === -1;
    });

    summarise(
      query === ""
        ? null
        : node.filter(function (d) {
            return haystack(d).indexOf(query) !== -1;
          }).size()
    );
  });

  // --- the panel ------------------------------------------------------------

  var selected = null;

  function open(id) {
    selected = id;
    node.classed("selected", function (d) { return d.id === id; });

    var n = byId.get(id).data;
    var into = [
      '<button type="button" class="close" id="panel-close">close</button>',
      "<h2>" + esc(n.id) + "</h2>",
      '<p class="muted">' + esc(badges(n)) + "</p>",
      "<section><h3>This node adds</h3>",
      n.constraint === null
        ? '<p class="constraint">Nothing — it is the root, and the root ' +
          "constrains nothing.</p>"
        : '<p class="constraint">' + esc(n.constraint) + "</p>",
      "</section>",
    ];

    if (n.constraints.length) {
      into.push(
        "<section><h3>In force here, root first</h3><ol>",
        n.constraints
          .map(function (c) { return "<li>" + esc(c) + "</li>"; })
          .join(""),
        "</ol></section>"
      );
    }

    into.push("<section><h3>Attempts</h3>", attemptsTable(n), "</section>");
    into.push(
      "<section><h3>Where the code came from</h3>",
      lineageList(n),
      "</section>"
    );

    panel.innerHTML = into.join("");
    panel.hidden = false;
    panel.scrollTop = 0;

    document.getElementById("panel-close").addEventListener("click", close);
  }

  function close() {
    selected = null;
    panel.hidden = true;
    node.classed("selected", false);
  }

  function badges(n) {
    var parts = ["depth " + n.depth];

    if (n.id === data.run.best_node_id) parts.push("holds the run's best");
    if (!n.attempts) parts.push("never attempted");
    if (n.dead) {
      parts.push("dead — " + n.stale + " attempts since its own best moved");
    }
    if (n.best !== null) {
      parts.push(
        "best " + fmt(n.best) + ", median " + fmt(n.median) +
          ", worst " + fmt(n.worst)
      );
    }
    if (n.last_worked !== null) parts.push("last worked at i" + n.last_worked);

    return parts.join(" · ");
  }

  function attemptsTable(n) {
    if (!n.solutions.length) {
      return (
        "<p>None. The iteration that created this node crashed before it " +
        "committed anything, so nothing was ever built for the idea.</p>"
      );
    }

    var rows = n.solutions.map(function (s) {
      var tags = Object.keys(s.tags)
        .sort()
        .map(function (k) { return k + "=" + s.tags[k]; })
        .join(" ");

      return (
        "<tr>" +
        "<td>" + (s.iteration === null ? "seed" : "i" + s.iteration) + "</td>" +
        '<td><span class="path">' + esc(s.id) + "</span></td>" +
        (s.score === null
          ? '<td class="failed">failed</td>'
          : '<td class="score">' + fmt(s.score) + "</td>") +
        "<td>" + esc(tags) + "</td>" +
        "</tr>" +
        '<tr><td colspan="4"><span class="path">' + esc(s.code_path) +
        (s.iteration_path === null ? "" : "  ·  " + esc(s.iteration_path)) +
        "</span></td></tr>"
      );
    });

    return (
      "<table><thead><tr><th>iter</th><th>solution</th><th>score</th>" +
      "<th>tags</th></tr></thead><tbody>" + rows.join("") + "</tbody></table>"
    );
  }

  function lineageList(n) {
    var lines = [];

    data.lineage.forEach(function (e) {
      if (e.to_node !== n.id) return;

      lines.push(
        "<li>i" + e.iteration + " started from <span class='path'>" +
          esc(e.from_solution) + "</span> at " + esc(e.from_node) +
          (e.cross_branch ? " — <strong>another branch</strong>" : "") +
          "</li>"
      );
    });

    data.lineage.forEach(function (e) {
      if (e.from_node !== n.id || e.to_node === n.id) return;

      lines.push(
        "<li>lent <span class='path'>" + esc(e.from_solution) + "</span> to " +
          esc(e.to_node) + " at i" + e.iteration +
          (e.cross_branch ? " — <strong>another branch</strong>" : "") +
          "</li>"
      );
    });

    return lines.length
      ? "<ol>" + lines.join("") + "</ol>"
      : "<p>Nothing. No solution here started from code elsewhere, and nothing " +
          "elsewhere started from code here.</p>";
  }

  function esc(text) {
    return String(text === null || text === undefined ? "" : text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // --- chrome ---------------------------------------------------------------

  document.getElementById("run-title").textContent = data.run.path;

  function summarise(matches) {
    var run = data.run;
    var parts = [
      data.nodes.length + (data.nodes.length === 1 ? " node" : " nodes"),
      run.iterations + (run.iterations === 1 ? " iteration" : " iterations"),
      run.solutions + (run.solutions === 1 ? " solution" : " solutions"),
    ];

    if (run.best_score !== null) {
      parts.push("best " + fmt(run.best_score) + " at " + run.best_node_id);
    }

    parts.push("rendered " + run.generated_at);

    if (matches !== null) parts.push(matches + " matching");

    document.getElementById("run-summary").textContent = parts.join("  ·  ");
  }

  summarise(null);

  var ranges = new Map();

  data.nodes.forEach(function (n) {
    if (n.band === null) return;

    var range = ranges.get(n.band);

    ranges.set(
      n.band,
      range
        ? [Math.min(range[0], n.best), Math.max(range[1], n.best)]
        : [n.best, n.best]
    );
  });

  var keys = Array.from(ranges.keys())
    .sort(d3.ascending)
    .map(function (band) {
      var range = ranges.get(band);
      var label =
        range[0] === range[1]
          ? fmt(range[0])
          : fmt(range[0]) + "–" + fmt(range[1]);

      return (
        '<span class="key"><span class="swatch" style="background: var(--band-' +
        rampStep(band) + ')"></span>' + (band === 0 ? "best " : "") + label +
        "</span>"
      );
    });

  document.getElementById("legend").innerHTML =
    '<span class="group"><strong>a node\'s best score</strong>' +
    (keys.length ? keys.join("") : "<span class='key'>nothing has scored</span>") +
    "</span>" +
    '<span class="group"><span class="key"><span class="swatch hollow"></span>' +
    "never attempted</span>" +
    '<span class="key"><span class="swatch ring"></span>run best</span>' +
    '<span class="key"><span class="rule"></span>lineage — the code a solution ' +
    "started from</span></span>";

  document.getElementById("theme").addEventListener("click", function () {
    var dark =
      document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);

    document.documentElement.dataset.theme = dark ? "light" : "dark";
  });

  window.addEventListener("keydown", function (event) {
    if (event.target === filterInput) return;

    if (event.key === "Escape" && selected !== null) close();
    if (event.key === "f") fit();
  });

  fit(false);
})();
