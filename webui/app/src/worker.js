// JSON API over the D1 database built by webui/prepdb.py. Everything else
// (public/index.html) is served by Workers static assets before this runs;
// the spectrogram images come straight from IMAGE_BASE_URL, not this Worker.
//
// Edits made in the app live in their own tables, hit_marks(results_id, mark),
// submodel_merges(child, parent) and type_settings(classification, enabled,
// min_llr, min_conf), so rebuilding display_set / linkages
// from web.db (npm run db:local / db:remote) keeps them.

const MARKS = ["noise", "wrong"];  // wrong = hit doesn't belong to its submodel

const LINKAGES = ["single", "complete", "average", "medoid"];
// Short browser cache: dedupes reloads, but a rebuilt DB / new column shows up within a minute.
const CACHE = { "Cache-Control": "public, max-age=60" };

const json = (body) => Response.json(body, { headers: CACHE });

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);

    if (pathname === "/api/config") {
      return json({ imageBase: env.IMAGE_BASE_URL, linkages: LINKAGES });
    }

    if (pathname === "/api/display_set") {
      const { results } = await env.DB.prepare(
        `SELECT d.results_id, d.submodel_id, d.wav_file, d.start_time_s, d.stop_time_s, d.region_offset_s,
                d.classification, d.classification_confidence, d.mean_llr_score,
                m.mark
         FROM display_set d LEFT JOIN hit_marks m ON m.results_id = d.results_id
         ORDER BY d.submodel_id, d.mean_llr_score DESC`).all();
      return json(results);
    }

    // Distances are symmetric, so only model_i < model_j is sent; all four
    // linkages in one query, since D1 bills rows scanned, not columns.
    if (pathname === "/api/linkages") {
      const rows = await env.DB.prepare(
        `SELECT model_i, model_j, ${LINKAGES.join(", ")} FROM linkages WHERE model_i < model_j`).raw();
      return json({ columns: ["model_i", "model_j", ...LINKAGES], rows });
    }

    // POST {"ids": [results_id, ...], "mark": "noise" | "wrong" | null}; null clears the mark.
    if (pathname === "/api/marks" && request.method === "POST") {
      const { ids, mark } = await request.json();
      if (mark !== null && !MARKS.includes(mark)) return new Response("bad mark", { status: 400 });
      const stmt = mark ? env.DB.prepare("INSERT OR REPLACE INTO hit_marks (results_id, mark) VALUES (?, ?)")
                        : env.DB.prepare("DELETE FROM hit_marks WHERE results_id = ?");
      await env.DB.batch(ids.filter(Number.isInteger).map((id) => (mark ? stmt.bind(id, mark) : stmt.bind(id))));
      return Response.json({ ok: true });
    }

    // Per-type filter settings: GET all rows, POST one {classification, enabled, min_llr, min_conf}.
    if (pathname === "/api/settings" && request.method === "GET") {
      return Response.json((await env.DB.prepare("SELECT * FROM type_settings").all()).results);
    }
    if (pathname === "/api/settings" && request.method === "POST") {
      const { classification, enabled, min_llr, min_conf } = await request.json();
      await env.DB.prepare(
        `INSERT OR REPLACE INTO type_settings (classification, enabled, min_llr, min_conf) VALUES (?, ?, ?, ?)`)
        .bind(classification, enabled ? 1 : 0, +min_llr, +min_conf).run();
      return Response.json({ ok: true });
    }

    // submodel_merges(child, parent): union-find parent pointers between submodel ids.
    if (pathname === "/api/merges" && request.method === "GET") {
      return Response.json(await env.DB.prepare("SELECT child, parent FROM submodel_merges").raw());
    }

    // POST {"op": "merge", "child": a, "parent": b}  links a's group under b's group root
    // POST {"op": "unlink", "node": m}                detaches m from its group, the rest stays merged
    if (pathname === "/api/merges" && request.method === "POST") {
      const { op, child, parent, node } = await request.json();
      const { results } = await env.DB.prepare("SELECT child, parent FROM submodel_merges").all();
      const up = new Map(results.map((r) => [r.child, r.parent]));
      const find = (m) => { while (up.has(m)) m = up.get(m); return m; };
      const stmts = [];
      if (op === "merge") {
        const a = find(child), b = find(parent);
        if (a !== b) stmts.push(env.DB.prepare("INSERT INTO submodel_merges (child, parent) VALUES (?, ?)").bind(a, b));
      } else if (op === "unlink") {
        const kids = results.filter((r) => r.parent === node).map((r) => r.child);
        if (up.has(node)) {  // children move up to node's parent
          stmts.push(env.DB.prepare("UPDATE submodel_merges SET parent = ? WHERE parent = ?").bind(up.get(node), node),
                     env.DB.prepare("DELETE FROM submodel_merges WHERE child = ?").bind(node));
        } else if (kids.length) {  // node is the root: its smallest child becomes the new root
          const root = Math.min(...kids);
          stmts.push(env.DB.prepare("DELETE FROM submodel_merges WHERE child = ?").bind(root),
                     env.DB.prepare("UPDATE submodel_merges SET parent = ? WHERE parent = ?").bind(root, node));
        }
      }
      if (stmts.length) await env.DB.batch(stmts);
      return Response.json({ ok: true });
    }

    // Clears all merges and all hit marks (type_settings are kept).
    if (pathname === "/api/reset" && request.method === "POST") {
      await env.DB.batch([env.DB.prepare("DELETE FROM submodel_merges"), env.DB.prepare("DELETE FROM hit_marks")]);
      return Response.json({ ok: true });
    }

    return new Response("not found", { status: 404 });
  },
};
