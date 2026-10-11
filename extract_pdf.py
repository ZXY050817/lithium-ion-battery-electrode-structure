import pymupdf
import sys

pdf_path = r"E:\chem\thesis.pdf"
output_path = r"E:\chem\thesis_full_text.txt"

doc = pymupdf.open(pdf_path)
print(f"Total pages: {doc.page_count}")

all_text = []
for i, page in enumerate(doc):
    text = page.get_text("text", sort=True)
    all_text.append(f"===== PAGE {i+1} =====\n{text}")

full_text = "\n\n".join(all_text)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"Extracted {len(full_text)} characters to {output_path}")
doc.close()
