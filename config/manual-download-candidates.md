# Manual-download candidates: Section 889 / C-SCRM

## Merged documents (retired individual PDFs)

The following individual PDFs were merged on 2026-09-01 to avoid 8 LLM calls per tiny page and
near-duplicate search results. The originals remain in `~/Documents/solicitation-library/` if needed.

- `Human-Centered Design Guide (Digital.gov).pdf` — merged from 16 digital.gov HCD guide pages:
  Introduction to human-centered design, HCD background, HCD principles, The HCD approach,
  Discovery concepts guide, Steps to discovery, Steps 1–8, Goals and insights, Methods
  (source: `~/Documents/solicitation-library/../digital.gov` downloads; live guide: https://digital.gov/guides/hcd/)
- `Federal Customer Experience Policy Package.pdf` — merged from 3 White House documents:
  Executive Order on Transforming Federal Customer Experience (EO 14058), "Putting the Public First"
  fact sheet, and Appendix III to OMB Circular No. A-130 (customer experience strategy)

## Section 889 / C-SCRM candidates

These documents are cited heavily across the solicitation library (18 of 46 solicitations reference
Section 889) but are only available as local PDFs, not at a stable public URL. The build pipeline
supports local ingestion: copy the PDFs below into `sources/__manual-download-gov-docs/` and they
will be processed automatically (category "Manually Downloaded", document name taken from the filename).

Source folder: `~/Documents/solicitation-library/889-compliance/`

| # | File | What it is |
|---|------|------------|
| 1 | `Section 889 - FAQs 30.pdf` | GSA/DOD FAQ compilation on Section 889 implementation |
| 2 | `GSA Enterprise-Level C-SCRM Strategic Plan 20210623_0.pdf` | GSA enterprise Cloud/SUPPLY CHAIN Risk Management strategic plan |
| 3 | `Final 889 Flyer updated_0.pdf` | GSA outreach flyer summarizing 889 obligations |
| 4 | `SCRM review board 889 PART A Rubric_20200901.pdf` | 889 Part A (interconnectivity) review rubric |
| 5 | `SCRM review board 889 PART B Rubric_20200901.pdf` | 889 Part B (equipment/services) review rubric |

Copy command (creates the target directory if needed):

```bash
mkdir -p sources/__manual-download-gov-docs
cp ~/Documents/solicitation-library/889-compliance/"Section 889 - FAQs 30.pdf" \
   ~/Documents/solicitation-library/889-compliance/"GSA Enterprise-Level C-SCRM Strategic Plan 20210623_0.pdf" \
   ~/Documents/solicitation-library/889-compliance/"Final 889 Flyer updated_0.pdf" \
   ~/Documents/solicitation-library/889-compliance/"SCRM review board 889 PART A Rubric_20200901.pdf" \
   ~/Documents/solicitation-library/889-compliance/"SCRM review board 889 PART B Rubric_20200901.pdf" \
   sources/__manual-download-gov-docs/
```

Related sources.csv entry added 2026-09-01: "Section 889 of the John S. McCain NDAA for FY2019
(Public Law 115-232)" → https://www.law.cornell.edu/uscode/text/41/8303 (the statute itself).
The PDFs above are the implementation guidance the statute's citations point to.

Also a candidate from the same library: `regulatory/far-companion.pdf` — only if you want the
procurement-process side (FAR) represented; it is out of the current web-standards mission.
