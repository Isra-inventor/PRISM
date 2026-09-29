"""A fake Europe PMC for tests (the real one is not reachable from CI).

Responses follow the real REST shapes: /search?format=json&resultType=core and
/{PMCID}/fullTextXML (JATS). The papers are invented test fixtures, not real
publications. Run as a server for browser tests:
    python tests/fake_europepmc.py 8090
    PRISM_EUROPEPMC_URL=http://127.0.0.1:8090 python -m uvicorn backend.main:app
"""

import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

PAPERS = [
    {"id": "90000001", "pmid": "90000001", "pmcid": "PMC9000001", "isOpenAccess": "Y",
     "title": "[TEST FIXTURE] Label-free quantification of plasma proteins with MaxQuant",
     "authorString": "Fixture A, Example B.", "pubYear": "2021",
     "journalInfo": {"journal": {"title": "Journal of Test Proteomics"}},
     "abstractText": ("We quantified plasma proteins by data-dependent acquisition and MaxQuant. "
                      "LFQ intensity values were used for all quantitative comparisons between groups. "
                      "iBAQ intensity was reported to estimate the relative abundance of proteins within a sample. "
                      "Peptide counts were used only as a quality control of identifications.")},
    {"id": "90000002", "pmid": "90000002", "pmcid": None, "isOpenAccess": "N",
     "title": "[TEST FIXTURE] iBAQ versus LFQ for absolute and relative protein abundance",
     "authorString": "Fixture C.", "pubYear": "2019",
     "journalInfo": {"journal": {"title": "Test Methods"}},
     "abstractText": ("Intensity based absolute quantification (iBAQ) divides the summed peptide intensity by the "
                      "number of theoretically observable peptides. iBAQ is suited to comparing proteins within a "
                      "sample, whereas LFQ intensity is normalized across samples and suited to comparing a protein "
                      "between samples.")},
    {"id": "90000003", "pmid": "90000003", "pmcid": "PMC9000003", "isOpenAccess": "Y",
     "title": "[TEST FIXTURE] Untargeted LC-MS metabolomics of serum: peak areas and QC samples",
     "authorString": "Fixture D, Example E.", "pubYear": "2022",
     "journalInfo": {"journal": {"title": "Test Metabolomics"}},
     "abstractText": ("Features were detected with MZmine and integrated peak area values were exported. "
                      "Peak area was analysed after normalization to pooled QC samples. "
                      "Blank injections were used to remove background features.")},
]

FULL_TEXT = {
    "PMC9000001": """<article><body>
<sec><title>Introduction</title><p>Plasma proteomics is widely used and this paragraph is not about methods at all, so it should be ignored.</p></sec>
<sec><title>Materials and methods</title>
<p>Raw files were processed with MaxQuant using the MaxLFQ algorithm. LFQ intensity values were log2 transformed and missing values were imputed from a normal distribution before statistical testing.</p>
<p>Proteins flagged as reverse hits or potential contaminants were removed prior to analysis, and <italic>iBAQ</italic> values were kept only to rank proteins by abundance within each sample.</p>
</sec></body></article>""",
    "PMC9000003": """<article><body>
<sec><title>Data processing</title>
<p>Peak area values were normalized by probabilistic quotient normalization using the pooled QC samples, and features with more than 30 percent missing values in QC samples were filtered out.</p>
</sec></body></article>""",
}


def _match(query):
    q = query.lower()
    words = [w.strip('"()') for w in q.replace(" and ", " ").replace(" or ", " ").split()]
    words = [w for w in words if w and ":" not in w and w not in ("has_abstract", "y")]
    hits = []
    for p in PAPERS:
        text = (p["title"] + " " + p["abstractText"]).lower()
        if sum(1 for w in words if w in text) >= max(1, len(words) // 2):
            hits.append(p)
    return hits


def respond(url):
    """URL -> bytes (or None for 404), like the real service."""
    u = urllib.parse.urlparse(url)
    if u.path.endswith("/search"):
        qs = urllib.parse.parse_qs(u.query)
        hits = _match(qs.get("query", [""])[0])[:int(qs.get("pageSize", ["25"])[0])]
        return json.dumps({"version": "6.9", "hitCount": len(hits),
                           "resultList": {"result": [dict(p, source="MED", inEPMC="Y") for p in hits]}}).encode()
    if u.path.endswith("/fullTextXML"):
        pmcid = u.path.split("/")[-2]
        x = FULL_TEXT.get(pmcid)
        return x.encode() if x else None
    return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = respond("http://x" + self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json" if self.path.split("?")[0].endswith("/search") else "application/xml")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8090), Handler).serve_forever()
