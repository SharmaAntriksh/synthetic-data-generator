/* ═══ data-preview.jsx — Dataset browser & table preview ═══ */

function DataPreview({ onBack }) {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [tables, setTables] = useState([]);
  const [tablesLoading, setTablesLoading] = useState(false);
  const [selectedTable, setSelectedTable] = useState(null);
  const [preview, setPreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [offset, setOffset] = useState(0);
  const [pageSize] = useState(100);
  const [searchTerm, setSearchTerm] = useState("");

  /* Load datasets on mount */
  useEffect(() => {
    fetch(API + "/datasets")
      .then(r => r.json())
      .then(data => { setDatasets(data.datasets || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  /* Load tables when a dataset is selected */
  useEffect(() => {
    if (!selectedDataset) { setTables([]); return; }
    setTablesLoading(true);
    setSelectedTable(null);
    setPreview(null);
    fetch(API + "/datasets/" + encodeURIComponent(selectedDataset.name) + "/tables")
      .then(r => r.json())
      .then(data => { setTables(data.tables || []); setTablesLoading(false); })
      .catch(() => setTablesLoading(false));
  }, [selectedDataset]);

  /* Load preview when table or offset changes */
  useEffect(() => {
    if (!selectedDataset || !selectedTable) return;
    setPreviewLoading(true);
    fetch(API + "/datasets/" + encodeURIComponent(selectedDataset.name) + "/tables/" + encodeURIComponent(selectedTable.name) + "/preview?offset=" + offset + "&limit=" + pageSize)
      .then(r => r.json())
      .then(data => { setPreview(data); setPreviewLoading(false); })
      .catch(() => setPreviewLoading(false));
  }, [selectedDataset, selectedTable, offset, pageSize]);

  const selectTable = (table) => {
    setSelectedTable(table);
    setOffset(0);
    setPreview(null);
  };

  const totalPages = preview ? Math.ceil(preview.total_rows / pageSize) : 0;
  const currentPage = Math.floor(offset / pageSize) + 1;

  const filteredTables = tables.filter(t =>
    !searchTerm || t.name.toLowerCase().includes(searchTerm.toLowerCase())
  );
  const dimTables = filteredTables.filter(t => t.category === "dimensions");
  const factTables = filteredTables.filter(t => t.category === "facts");

  /* Format helpers */
  const fmtBadge = (fmt) => {
    const colors = { csv: "#e67e22", parquet: "#3498db", delta: "#2ecc71", unknown: "#95a5a6" };
    return { background: colors[fmt] || colors.unknown, color: "#fff", padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".04em" };
  };

  return (
    <div>
      <button onClick={onBack} style={{display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--dim)", fontSize: 12.5, cursor: "pointer", fontFamily: "var(--sans)", marginBottom: 18, transition: "all .12s"}} onMouseOver={e => e.currentTarget.style.borderColor = "var(--accent)"} onMouseOut={e => e.currentTarget.style.borderColor = "var(--border)"}>
        {"\u2190"} Back to configuration
      </button>
      <h1 style={{fontSize: 22, fontWeight: 700, letterSpacing: "-.02em", marginBottom: 4}}>Data Preview</h1>
      <p style={{fontSize: 13, color: "var(--dim)", marginTop: 3, marginBottom: 22}}>Browse generated datasets and preview table contents</p>

      {/* Dataset selector */}
      {loading ? (
        <div style={{padding: 40, textAlign: "center", color: "var(--muted)"}}>Loading datasets...</div>
      ) : datasets.length === 0 ? (
        <div style={{padding: 40, textAlign: "center", color: "var(--muted)"}}>No generated datasets found. Run a generation first.</div>
      ) : (
        <>
          <div style={{marginBottom: 20}}>
            <div style={{fontSize: 10.5, fontWeight: 700, color: "var(--muted)", letterSpacing: ".09em", textTransform: "uppercase", marginBottom: 8}}>Select Dataset</div>
            <div style={{display: "flex", flexDirection: "column", gap: 6}}>
              {datasets.map(ds => {
                const isSelected = selectedDataset && selectedDataset.name === ds.name;
                return (
                  <button key={ds.name} onClick={() => setSelectedDataset(ds)} style={{
                    display: "flex", alignItems: "center", gap: 12, width: "100%", padding: "10px 14px", borderRadius: 8,
                    cursor: "pointer", border: `1px solid ${isSelected ? "var(--accent)" : "var(--border)"}`,
                    background: isSelected ? "var(--glow)" : "var(--surface)", fontFamily: "var(--sans)", transition: "all .12s", textAlign: "left",
                  }} onMouseOver={e => { if (!isSelected) e.currentTarget.style.borderColor = "var(--accent)"; }}
                     onMouseOut={e => { if (!isSelected) e.currentTarget.style.borderColor = "var(--border)"; }}>
                    <span style={{flex: 1, fontSize: 12.5, fontWeight: isSelected ? 600 : 400, color: isSelected ? "var(--accent)" : "var(--text)"}}>{ds.name}</span>
                    <span style={fmtBadge(ds.format)}>{ds.format}</span>
                    <span style={{fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)"}}>{ds.table_count} tables</span>
                    <span style={{fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)"}}>{ds.size_mb} MB</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Tables list + Preview */}
          {selectedDataset && (
            <div style={{display: "flex", gap: 20, minHeight: 400}}>
              {/* Table list panel */}
              <div style={{width: 240, flexShrink: 0}}>
                <div style={{fontSize: 10.5, fontWeight: 700, color: "var(--muted)", letterSpacing: ".09em", textTransform: "uppercase", marginBottom: 8}}>Tables</div>
                <input
                  type="text" placeholder="Filter tables..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)}
                  style={{width: "100%", padding: "7px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)", fontSize: 12, fontFamily: "var(--sans)", marginBottom: 10, boxSizing: "border-box", outline: "none"}}
                />
                {tablesLoading ? (
                  <div style={{padding: 20, textAlign: "center", color: "var(--muted)", fontSize: 12}}>Loading...</div>
                ) : (
                  <div style={{maxHeight: 600, overflowY: "auto"}}>
                    {dimTables.length > 0 && <>
                      <div style={{fontSize: 10, fontWeight: 600, color: "var(--muted)", padding: "6px 0 4px", textTransform: "uppercase", letterSpacing: ".06em"}}>Dimensions ({dimTables.length})</div>
                      {dimTables.map(t => {
                        const isSel = selectedTable && selectedTable.name === t.name;
                        return (
                          <button key={t.name} onClick={() => selectTable(t)} style={{
                            display: "block", width: "100%", textAlign: "left", padding: "6px 10px", borderRadius: 6,
                            border: "none", background: isSel ? "var(--glow)" : "transparent",
                            color: isSel ? "var(--accent)" : "var(--text)", fontSize: 12, cursor: "pointer",
                            fontFamily: "var(--mono)", fontWeight: isSel ? 600 : 400, transition: "all .1s",
                          }} onMouseOver={e => { if (!isSel) e.currentTarget.style.background = "var(--alt)"; }}
                             onMouseOut={e => { if (!isSel) e.currentTarget.style.background = isSel ? "var(--glow)" : "transparent"; }}>
                            {t.name}
                          </button>
                        );
                      })}
                    </>}
                    {factTables.length > 0 && <>
                      <div style={{fontSize: 10, fontWeight: 600, color: "var(--muted)", padding: "10px 0 4px", textTransform: "uppercase", letterSpacing: ".06em"}}>Facts ({factTables.length})</div>
                      {factTables.map(t => {
                        const isSel = selectedTable && selectedTable.name === t.name;
                        return (
                          <button key={t.name} onClick={() => selectTable(t)} style={{
                            display: "block", width: "100%", textAlign: "left", padding: "6px 10px", borderRadius: 6,
                            border: "none", background: isSel ? "var(--glow)" : "transparent",
                            color: isSel ? "var(--accent)" : "var(--text)", fontSize: 12, cursor: "pointer",
                            fontFamily: "var(--mono)", fontWeight: isSel ? 600 : 400, transition: "all .1s",
                          }} onMouseOver={e => { if (!isSel) e.currentTarget.style.background = "var(--alt)"; }}
                             onMouseOut={e => { if (!isSel) e.currentTarget.style.background = isSel ? "var(--glow)" : "transparent"; }}>
                            {t.name}
                          </button>
                        );
                      })}
                    </>}
                  </div>
                )}
              </div>

              {/* Preview panel */}
              <div style={{flex: 1, minWidth: 0}}>
                {!selectedTable ? (
                  <div style={{padding: 40, textAlign: "center", color: "var(--muted)", fontSize: 13}}>Select a table to preview its data</div>
                ) : previewLoading && !preview ? (
                  <div style={{padding: 40, textAlign: "center", color: "var(--muted)", fontSize: 13}}>Loading preview...</div>
                ) : preview ? (
                  <div>
                    {/* Table header */}
                    <div style={{display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12}}>
                      <div style={{display: "flex", alignItems: "center", gap: 10}}>
                        <span style={{fontSize: 15, fontWeight: 700, color: "var(--text)"}}>{preview.table}</span>
                        <span style={fmtBadge(preview.format)}>{preview.format}</span>
                        <span style={{fontSize: 12, color: "var(--muted)"}}>{preview.category}</span>
                      </div>
                      <div style={{display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "var(--dim)"}}>
                        <span style={{fontFamily: "var(--mono)"}}>{preview.total_rows.toLocaleString()} rows</span>
                        <span style={{color: "var(--muted)"}}>{"\u00B7"}</span>
                        <span style={{fontFamily: "var(--mono)"}}>{preview.columns.length} cols</span>
                      </div>
                    </div>

                    {/* Data table */}
                    <div style={{border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden"}}>
                      <div style={{overflowX: "auto", maxHeight: 480}}>
                        <table style={{width: "100%", borderCollapse: "collapse", fontSize: 11.5, fontFamily: "var(--mono)"}}>
                          <thead>
                            <tr style={{position: "sticky", top: 0, zIndex: 1}}>
                              <th style={{padding: "8px 10px", background: "var(--alt)", borderBottom: "2px solid var(--border)", fontWeight: 700, color: "var(--dim)", textAlign: "right", fontSize: 10, width: 50, whiteSpace: "nowrap"}}>#</th>
                              {preview.columns.map(col => (
                                <th key={col} style={{padding: "8px 10px", background: "var(--alt)", borderBottom: "2px solid var(--border)", fontWeight: 600, color: "var(--text)", textAlign: "left", whiteSpace: "nowrap"}}>{col}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {preview.rows.map((row, rowIdx) => (
                              <tr key={rowIdx} style={{borderBottom: "1px solid var(--border)"}} onMouseOver={e => e.currentTarget.style.background = "var(--glow)"} onMouseOut={e => e.currentTarget.style.background = "transparent"}>
                                <td style={{padding: "5px 10px", color: "var(--muted)", textAlign: "right", fontSize: 10, whiteSpace: "nowrap"}}>{offset + rowIdx + 1}</td>
                                {row.map((cell, colIdx) => (
                                  <td key={colIdx} style={{padding: "5px 10px", color: "var(--text)", whiteSpace: "nowrap", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis"}}>{cell === "" || cell === null ? <span style={{color: "var(--muted)", fontStyle: "italic"}}>null</span> : String(cell)}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Pagination */}
                    {totalPages > 1 && (
                      <div style={{display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 12, padding: "8px 0"}}>
                        <div style={{display: "flex", gap: 6}}>
                          <button onClick={() => setOffset(0)} disabled={offset === 0} style={{padding: "5px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", color: offset === 0 ? "var(--muted)" : "var(--text)", fontSize: 12, cursor: offset === 0 ? "default" : "pointer", fontFamily: "var(--sans)", opacity: offset === 0 ? 0.5 : 1}}>First</button>
                          <button onClick={() => setOffset(Math.max(0, offset - pageSize))} disabled={offset === 0} style={{padding: "5px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", color: offset === 0 ? "var(--muted)" : "var(--text)", fontSize: 12, cursor: offset === 0 ? "default" : "pointer", fontFamily: "var(--sans)", opacity: offset === 0 ? 0.5 : 1}}>Prev</button>
                        </div>
                        <span style={{fontSize: 12, color: "var(--dim)", fontFamily: "var(--mono)"}}>
                          Page {currentPage} of {totalPages.toLocaleString()} {previewLoading && <span style={{color: "var(--muted)"}}>loading...</span>}
                        </span>
                        <div style={{display: "flex", gap: 6}}>
                          <button onClick={() => setOffset(offset + pageSize)} disabled={offset + pageSize >= preview.total_rows} style={{padding: "5px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", color: offset + pageSize >= preview.total_rows ? "var(--muted)" : "var(--text)", fontSize: 12, cursor: offset + pageSize >= preview.total_rows ? "default" : "pointer", fontFamily: "var(--sans)", opacity: offset + pageSize >= preview.total_rows ? 0.5 : 1}}>Next</button>
                          <button onClick={() => setOffset((totalPages - 1) * pageSize)} disabled={offset + pageSize >= preview.total_rows} style={{padding: "5px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", color: offset + pageSize >= preview.total_rows ? "var(--muted)" : "var(--text)", fontSize: 12, cursor: offset + pageSize >= preview.total_rows ? "default" : "pointer", fontFamily: "var(--sans)", opacity: offset + pageSize >= preview.total_rows ? 0.5 : 1}}>Last</button>
                        </div>
                      </div>
                    )}
                  </div>
                ) : null}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
