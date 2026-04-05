import os
import shutil
import re
import glob

base_dir = '/users/PAS2030/srishti/asr_fairness'
paper_dir = os.path.join(base_dir, 'paper')
out_dir = os.path.join(base_dir, 'overleaf_new')

if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir)

# Create figures and tables dirs
figs_dir = os.path.join(out_dir, 'figures')
tables_dir = os.path.join(out_dir, 'tables')
os.makedirs(figs_dir)
os.makedirs(tables_dir)

# Copy root paper files
for f in os.listdir(paper_dir):
    if f.endswith(('.tex', '.bib', '.bst', '.sty')):
        shutil.copy2(os.path.join(paper_dir, f), os.path.join(out_dir, f))

# Consolidate figures
# \graphicspath in paper:
#   {../results/figures/}
#   {../results/hallucination_analysis/}
#   {../results/commonvoice/analysis/}
#   {../results/fairspeech/analysis/}
for search_dir in [
    'results/figures',
    'results/hallucination_analysis',
    'results/commonvoice/analysis',
    'results/fairspeech/analysis'
]:
    d = os.path.join(base_dir, search_dir)
    if os.path.exists(d):
        for pdf in glob.glob(os.path.join(d, '*.pdf')):
            shutil.copy2(pdf, os.path.join(figs_dir, os.path.basename(pdf)))

# Process .tex files to rewrite paths
for tex_file in ['colm2026_conference.tex', 'appendix.tex']:
    tex_path = os.path.join(out_dir, tex_file)
    if not os.path.exists(tex_path):
        continue
    with open(tex_path, 'r') as f:
        content = f.read()
    
    # 1. Replace \graphicspath{...} with \graphicspath{{figures/}}
    content = re.sub(r'\\graphicspath\{[\s\S]*?\}', r'\\graphicspath{{figures/}}', content)
    
    # 2. Find and replace \input{../results/.../name.tex} -> \input{tables/name.tex}
    # and copy the actual file to tables/
    def replace_input(match):
        orig_path = match.group(1)
        if '../results/' in orig_path:
            full_path = os.path.normpath(os.path.join(paper_dir, orig_path))
            if orig_path.endswith('.tex'):
               pass # already has .tex
            else:
               full_path += '.tex'
               
            if os.path.exists(full_path):
                basename = os.path.basename(full_path)
                shutil.copy2(full_path, os.path.join(tables_dir, basename))
                return f'\\input{{tables/{basename}}}'
        return match.group(0)

    content = re.sub(r'\\input\{([^}]+)\}', replace_input, content)

    # Some graphics might be \includegraphics{../results/.../fig.pdf}
    def replace_includegraphics(match):
        pre = match.group(1)
        orig_path = match.group(2)
        if '../results/' in orig_path:
            basename = os.path.basename(orig_path)
            return f'\\includegraphics{pre}{{{basename}}}'
        return match.group(0)
        
    content = re.sub(r'\\includegraphics(\[.*?\])?\{([^}]+)\}', replace_includegraphics, content)

    with open(tex_path, 'w') as f:
        f.write(content)

print(f"Successfully processed files into {out_dir}")
