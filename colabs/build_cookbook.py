"""Builds colabs/cancer_variant_cookbook.ipynb from the cell sources below.

Edit this file, then run ``python colabs/build_cookbook.py``. Do not edit the
.ipynb by hand: the cell sources here are raw strings, so escapes such as the
``\n`` in ``ylabel_template`` stay escapes in the notebook.

Cell code follows the AlphaGenome docs notebooks ``variant_scoring_ui`` and
``splicing_variant_scoring``. Model version, output types, tissue and strand
rule match the sidecar (``app/scoring.py``).
"""

from __future__ import annotations

from pathlib import Path

import nbformat

ALPHAGENOME_VERSION = "0.10.0"
NOTEBOOK_PATH = Path(__file__).with_name("cancer_variant_cookbook.ipynb")
# Branch that hosts the notebook on GitHub. Colab opens it from there.
# Local demo: the working branch. Change to "main" after the merge.
NOTEBOOK_REF = "w6-muse-copilot"
COLAB_BADGE_URL = (
    "https://colab.research.google.com/github/DixitPatel/alphagenome_sidecar"
    f"/blob/{NOTEBOOK_REF}/colabs/cancer_variant_cookbook.ipynb"
)

INTRO = f"""# AlphaGenome variant viewer for cBioPortal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_BADGE_URL})

This notebook plots AlphaGenome predictions for one somatic variant from the
cBioPortal patient view. It uses the same model version, output types, tissue
and strand rule as the cBioPortal AlphaGenome sidecar.

**Steps**

1. Add your AlphaGenome API key as the Colab secret `ALPHA_GENOME_API_KEY`
   (key icon in the left panel). Get a key at https://alphagenome.google/api.
2. Run cells 1 and 2.
3. In cBioPortal, open **Ask Muse > Colab Cookbook**, pick a variant and click
   **Copy cell 3**. Here, select all text in cell 3 and paste.
4. Run cells 3 to 6.

The default values in cell 3 are KRAS G12D (GRCh38 chr12:25245350 C>T) in
pancreas (UBERON:0001264).

Model predictions for research use only. Not for clinical decisions. Use of the
AlphaGenome API is subject to https://alphagenome.google/terms.
"""

CELL_1 = f"""# @title 1. Install AlphaGenome (same version as the cBioPortal sidecar)
from IPython.display import clear_output
! pip install alphagenome=={ALPHAGENOME_VERSION}
clear_output()
"""

CELL_2 = r"""# @title 2. Imports, model and gene annotation
import json

import pandas as pd
from IPython.display import display
from alphagenome import colab_utils
from alphagenome.data import gene_annotation, genome, transcript
from alphagenome.models.v1 import dna_client
from alphagenome.visualization import plot_components

# Reads the Colab secret ALPHA_GENOME_API_KEY.
dna_model = dna_client.create(colab_utils.get_api_key())

HG38_GTF_FEATHER = (
    'https://storage.googleapis.com/alphagenome/reference/gencode/'
    'hg38/gencode.v46.annotation.gtf.gz.feather'
)
gtf = pd.read_feather(HG38_GTF_FEATHER)
gtf_transcript = gene_annotation.filter_transcript_support_level(
    gene_annotation.filter_protein_coding(gtf), ['1']
)
longest_transcript_extractor = transcript.TranscriptExtractor(
    gene_annotation.filter_to_longest_transcript(gtf_transcript)
)
"""

# Ask Muse renders this cell with the selected variant (app/colab.py).
CELL_3 = r"""# @title 3. Variant and tissue (paste the block from Ask Muse)
variant_chromosome = 'chr12'  # @param {type:"string"}
variant_position = 25245350  # @param {type:"integer"}
variant_reference_bases = 'C'  # @param {type:"string"}
variant_alternate_bases = 'T'  # @param {type:"string"}
gene_symbol = 'KRAS'  # @param {type:"string"}
ontology_terms = ['UBERON:0001264']  # @param
sidecar_scores = ''  # @param {type:"string"}
"""

CELL_4 = r"""# @title 4. Predict REF and ALT tracks (1 Mb window)
variant = genome.Variant(
    chromosome=variant_chromosome,
    position=variant_position,
    reference_bases=variant_reference_bases,
    alternate_bases=variant_alternate_bases,
)
interval = variant.reference_interval.resize(dna_client.SEQUENCE_LENGTH_1MB)

# The same four output types the sidecar scores.
output = dna_model.predict_variant(
    interval=interval,
    variant=variant,
    organism=dna_client.Organism.HOMO_SAPIENS,
    requested_outputs=[
        dna_client.OutputType.RNA_SEQ,
        dna_client.OutputType.SPLICE_SITES,
        dna_client.OutputType.SPLICE_SITE_USAGE,
        dna_client.OutputType.SPLICE_JUNCTIONS,
    ],
    ontology_terms=ontology_terms,
)

# Keep tracks on the gene strand plus unstranded tracks, as the sidecar does
# (tidy_scores with match_gene_strand=True).
gene_strand = gene_annotation.get_gene_interval(gtf, gene_symbol=gene_symbol).strand


def on_gene_strand(tdata):
  if tdata is None:
    return None
  if gene_strand == '+':
    return tdata.filter_to_nonnegative_strand()
  return tdata.filter_to_nonpositive_strand()


ref, alt = output.reference, output.alternate
print(f'{gene_symbol} is on the {gene_strand} strand. Window: {interval}')
"""

CELL_5 = r"""# @title 5. Sidecar scores (the same Mongo record as the Mutations table)
if sidecar_scores.strip():
  scores = json.loads(sidecar_scores)
  print(
      f"score_level: {scores.get('score_level')}  "
      f"merged_splicing_score: {scores.get('merged_splicing_score')}"
  )
  display(pd.DataFrame(scores.get('top', [])))
else:
  print('No sidecar scores. Paste the block from Ask Muse into cell 3.')
"""

CELL_6 = r"""# @title 6. Plot REF vs ALT { run: "auto" }
plot_rna_seq = True  # @param {type:"boolean"}
plot_splice_sites = True  # @param {type:"boolean"}
plot_splice_site_usage = True  # @param {type:"boolean"}
plot_splice_junctions = True  # @param {type:"boolean"}
ref_color = 'dimgrey'  # @param {type:"string"}
alt_color = 'red'  # @param {type:"string"}
plot_interval_width = 43008  # @param {type:"slider", min:2048, max:1048576, step:2048}
plot_interval_shift = 0  # @param {type:"slider", min:-524288, max:524288, step:2048}

ref_alt_colors = {'REF': ref_color, 'ALT': alt_color}
components = [
    plot_components.TranscriptAnnotation(
        longest_transcript_extractor.extract(interval)
    )
]

overlaid = [
    (plot_rna_seq, 'RNA_SEQ', ref.rna_seq, alt.rna_seq,
     'RNA_SEQ: {biosample_name} ({strand})\n{name}'),
    (plot_splice_sites, 'SPLICE_SITES', ref.splice_sites, alt.splice_sites,
     'SPLICE_SITES: {name} ({strand})'),
    (plot_splice_site_usage, 'SPLICE_SITE_USAGE', ref.splice_site_usage,
     alt.splice_site_usage,
     'SPLICE_SITE_USAGE: {biosample_name} ({strand})\n{name}'),
]
for enabled, output_type, ref_data, alt_data, ylabel_template in overlaid:
  if not enabled:
    continue
  ref_data, alt_data = on_gene_strand(ref_data), on_gene_strand(alt_data)
  if ref_data is None or ref_data.values.shape[-1] == 0:
    print(f'No {output_type} tracks for this tissue and strand.')
    continue
  components.append(
      plot_components.OverlaidTracks(
          tdata={'REF': ref_data, 'ALT': alt_data},
          colors=ref_alt_colors,
          ylabel_template=ylabel_template,
      )
  )

if plot_splice_junctions and ref.splice_junctions is not None:
  ref_junctions = ref.splice_junctions.filter_to_strand(gene_strand)
  alt_junctions = alt.splice_junctions.filter_to_strand(gene_strand)
  junction_template = 'SPLICE_JUNCTIONS: {biosample_name} ({strand})\n{name}'
  components.append(
      plot_components.Sashimi(
          ref_junctions, ylabel_template='REF: ' + junction_template
      )
  )
  components.append(
      plot_components.Sashimi(
          alt_junctions, ylabel_template='ALT: ' + junction_template
      )
  )

if plot_interval_width > interval.width:
  raise ValueError(
      f'plot_interval_width ({plot_interval_width}) must be <= '
      f'interval.width ({interval.width}).'
  )

plot = plot_components.plot(
    components=components,
    interval=interval.shift(plot_interval_shift).resize(plot_interval_width),
    annotations=[plot_components.VariantAnnotation([variant])],
    title=f'{gene_symbol} {variant}: REF vs ALT',
)
"""

CODE_CELLS = (CELL_1, CELL_2, CELL_3, CELL_4, CELL_5, CELL_6)


def build_notebook() -> nbformat.NotebookNode:
    """Returns the cookbook notebook. Cell ids are fixed so builds are stable."""
    cells = [nbformat.v4.new_markdown_cell(INTRO.strip(), id="intro")]
    for number, source in enumerate(CODE_CELLS, start=1):
        cells.append(
            nbformat.v4.new_code_cell(
                source.strip(),
                id=f"cell-{number}",
                metadata={"cellView": "form"},
            )
        )
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "colab": {"name": "AlphaGenome variant viewer for cBioPortal", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return notebook


def render() -> str:
    notebook = build_notebook()
    nbformat.validate(notebook)
    return nbformat.writes(notebook) + "\n"


if __name__ == "__main__":
    NOTEBOOK_PATH.write_text(render(), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")
