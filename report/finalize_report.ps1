# Opens the generated report in Word, fills the Table of Contents and the lists of
# figures/tables (page numbers are only known after layout), saves, and exports the PDF.
param([string]$Dir = $PSScriptRoot)
$name = "Manufacturing_Analytics_Project_Report_Bearing_Prognostics"
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$doc = $w.Documents.Open("$Dir\$name.docx")
# compact built-in TOC styles (-20 = TOC 1, -21 = TOC 2): 10 pt, single exact spacing
foreach ($id in -20, -21) {
    $st = $doc.Styles.Item($id)
    $st.Font.Name = "Times New Roman"; $st.Font.Size = 10
    $st.ParagraphFormat.SpaceBefore = 0; $st.ParagraphFormat.SpaceAfter = 0
    $st.ParagraphFormat.LineSpacingRule = 4; $st.ParagraphFormat.LineSpacing = 11.5
}
$doc.Styles.Item(-20).ParagraphFormat.SpaceBefore = 0
foreach ($t in $doc.TablesOfContents) { $t.Update() }
$doc.Fields.Update() | Out-Null
foreach ($t in $doc.TablesOfContents) { $t.Update() }   # second pass: lists may shift pages
# the empty paragraph that closes each TOC field must not spill onto a new page
foreach ($t in $doc.TablesOfContents) {
    $end = $t.Range.Paragraphs.Last.Range
    $end.Font.Size = 1
    $end.ParagraphFormat.SpaceBefore = 0; $end.ParagraphFormat.SpaceAfter = 0
    $end.ParagraphFormat.LineSpacingRule = 4; $end.ParagraphFormat.LineSpacing = 1
}
$doc.Save()
$doc.SaveAs([ref]"$Dir\$name.pdf", [ref]17)
"pages: " + $doc.ComputeStatistics(2)
$doc.Close(0)
$w.Quit()
