// Optional vector assembly; see README.md for the compile command.
#set page(
  width: 290.7mm,
  height: 141.2mm,
  margin: 1mm,
  fill: white,
)

#set text(font: "Arial", size: 12pt, fill: black)

#let model-title(body) = align(center, text(size: 20pt)[#body])
#let panel(path) = box(width: 100%, height: 100%, inset: 0pt)[
  #image(path, width: 100%, height: 100%, fit: "contain")
]

#grid(
  columns: (95.9mm, 95.9mm, 95.9mm),
  rows: (9mm, 65mm, 65mm),
  column-gutter: 0.5mm,
  row-gutter: 0.1mm,
  align: center + horizon,

  model-title([TFIM]),
  model-title([$Z_2$]),
  model-title([XXZ]),

  panel("figures/tfim_parameter.pdf"),
  panel("figures/z2_parameter.pdf"),
  panel("figures/xxz_parameter.pdf"),

  panel("figures/tfim_latent.pdf"),
  panel("figures/z2_latent.pdf"),
  panel("figures/xxz_latent.pdf"),
)
