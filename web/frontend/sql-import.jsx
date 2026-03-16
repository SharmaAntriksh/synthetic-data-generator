/* ═══ sql-import.jsx — SQL Server Import UI ═══ */

function SqlImport({ onBack }) {
  /* Connection settings */
  const [server, setServer] = useState("");
  const [database, setDatabase] = useState("");
  const [trusted, setTrusted] = useState(true);
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [applyCci, setApplyCci] = useState(false);
  const [odbcDriver, setOdbcDriver] = useState("");

  /* Dataset selection */
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [datasetsLoading, setDatasetsLoading] = useState(true);

  /* ODBC driver info */
  const [drivers, setDrivers] = useState([]);
  const [driverAvailable, setDriverAvailable] = useState(null);
  const [driverError, setDriverError] = useState(null);

  /* Import job state */
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([]);
  const [elapsed, setElapsed] = useState(0);
  const [jobStatus, setJobStatus] = useState(null); /* done | failed | cancelled */
  const timerRef = useRef(null);
  const logsEndRef = useRef(null);

  /* Load datasets and check ODBC on mount */
  useEffect(() => {
    fetch(API + "/datasets")
      .then(r => r.json())
      .then(data => {
        /* Only show CSV datasets (SQL import requires CSV format) */
        const csvDatasets = (data.datasets || []).filter(ds => ds.format === "csv");
        setDatasets(csvDatasets);
        setDatasetsLoading(false);
      })
      .catch(() => setDatasetsLoading(false));

    fetch(API + "/import/drivers")
      .then(r => r.json())
      .then(data => {
        setDriverAvailable(data.available && data.drivers.length > 0);
        setDrivers(data.drivers || []);
        if (data.drivers && data.drivers.length > 0) {
          setOdbcDriver(data.drivers[0]);
        }
        if (data.error) setDriverError(data.error);
      })
      .catch(() => setDriverAvailable(false));
  }, []);

  /* Auto-scroll logs */
  useEffect(() => {
    if (logsEndRef.current) logsEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  /* Start import */
  const startImport = () => {
    if (!server.trim() || !database.trim() || !selectedDataset) return;
    if (!trusted && !user.trim()) return;

    setLogs([]);
    setJobStatus(null);
    setIsRunning(true);
    setElapsed(0);

    const startTime = Date.now();
    timerRef.current = setInterval(() => setElapsed((Date.now() - startTime) / 1000), 100);

    fetch(API + "/import/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        server: server.trim(),
        database: database.trim(),
        dataset: selectedDataset,
        trusted,
        user: trusted ? null : user.trim(),
        password: trusted ? null : password,
        apply_cci: applyCci,
        odbc_driver: odbcDriver || null,
      }),
    })
      .then(r => {
        if (!r.ok) return r.json().then(d => { throw new Error(d.detail || "Failed to start import"); });
        return r.json();
      })
      .then(() => {
        /* Stream logs via SSE */
        const es = new EventSource(API + "/import/stream");
        es.onmessage = (ev) => {
          const data = JSON.parse(ev.data);
          if (data.type === "log") setLogs(prev => [...prev, data.line]);
          if (data.type === "status") setElapsed(data.elapsed);
          if (data.type === "end" || data.type === "idle") {
            es.close();
            clearInterval(timerRef.current);
            setIsRunning(false);
            setElapsed(data.elapsed || 0);
            setJobStatus(data.status || "done");
          }
        };
        es.onerror = () => {
          es.close();
          clearInterval(timerRef.current);
          setIsRunning(false);
          setJobStatus("failed");
        };
      })
      .catch(err => {
        clearInterval(timerRef.current);
        setIsRunning(false);
        setJobStatus("failed");
        setLogs(prev => [...prev, `ERROR: ${err.message}`]);
      });
  };

  /* Cancel import */
  const cancelImport = () => {
    fetch(API + "/import/cancel", { method: "POST" }).catch(() => {});
  };

  const canStart = server.trim() && database.trim() && selectedDataset && (trusted || user.trim()) && !isRunning;

  const inputStyle = { width: "100%", padding: "8px 10px", borderRadius: 6, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)", fontSize: 12.5, fontFamily: "var(--sans)", boxSizing: "border-box", outline: "none" };
  const labelStyle = { fontSize: 11.5, fontWeight: 600, color: "var(--dim)", marginBottom: 4, display: "block" };

  return (
    <div>
      <button onClick={onBack} style={{display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 14px", borderRadius: 7, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--dim)", fontSize: 12.5, cursor: "pointer", fontFamily: "var(--sans)", marginBottom: 18, transition: "all .12s"}} onMouseOver={e => e.currentTarget.style.borderColor = "var(--accent)"} onMouseOut={e => e.currentTarget.style.borderColor = "var(--border)"}>
        {"\u2190"} Back to configuration
      </button>
      <h1 style={{fontSize: 22, fontWeight: 700, letterSpacing: "-.02em", marginBottom: 4}}>SQL Server Import</h1>
      <p style={{fontSize: 13, color: "var(--dim)", marginTop: 3, marginBottom: 22}}>Import generated CSV data into SQL Server</p>

      {/* ODBC driver warning */}
      {driverAvailable === false && (
        <div style={{padding: "10px 14px", marginBottom: 16, borderRadius: 8, background: "var(--errBg)", border: "1px solid rgba(239,68,68,.15)", fontSize: 12.5, color: "var(--err)", display: "flex", alignItems: "flex-start", gap: 8}}>
          <span style={{fontWeight: 700, flexShrink: 0}}>&#10005;</span>
          <div>
            <strong>ODBC Driver not found.</strong> SQL Server import requires an ODBC driver.
            {driverError && <div style={{marginTop: 4, fontSize: 11.5, opacity: .8}}>{driverError}</div>}
            <div style={{marginTop: 6, fontSize: 11}}>
              Install from: <a href="https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server" target="_blank" rel="noopener" style={{color: "var(--accent)"}}>Microsoft ODBC Driver for SQL Server</a>
            </div>
          </div>
        </div>
      )}

      {driverAvailable && (
        <div style={{padding: "10px 14px", marginBottom: 16, borderRadius: 8, background: "rgba(34,197,94,.06)", border: "1px solid rgba(34,197,94,.12)", fontSize: 12, color: "var(--ok)", display: "flex", alignItems: "center", gap: 8}}>
          <span>&#10003;</span>
          <span>ODBC Driver: <strong>{drivers[0]}</strong>{drivers.length > 1 ? ` (+${drivers.length - 1} more)` : ""}</span>
        </div>
      )}

      <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24}}>
        {/* Left column: Connection */}
        <div>
          <div style={{fontSize: 10.5, fontWeight: 700, color: "var(--muted)", letterSpacing: ".09em", textTransform: "uppercase", marginBottom: 12}}>Connection</div>

          <div style={{marginBottom: 12}}>
            <label style={labelStyle}>Server</label>
            <input type="text" value={server} onChange={e => setServer(e.target.value)} placeholder="HOSTNAME\INSTANCE" style={inputStyle} disabled={isRunning} />
          </div>

          <div style={{marginBottom: 12}}>
            <label style={labelStyle}>Database</label>
            <input type="text" value={database} onChange={e => setDatabase(e.target.value)} placeholder="ContosoRetailDW" style={inputStyle} disabled={isRunning} />
          </div>

          <div style={{marginBottom: 12}}>
            <label style={{...labelStyle, display: "flex", alignItems: "center", gap: 8, cursor: "pointer"}}>
              <input type="checkbox" checked={trusted} onChange={e => setTrusted(e.target.checked)} disabled={isRunning} />
              Windows Authentication (Trusted Connection)
            </label>
          </div>

          {!trusted && (
            <div style={{display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12}}>
              <div>
                <label style={labelStyle}>Username</label>
                <input type="text" value={user} onChange={e => setUser(e.target.value)} placeholder="sa" style={inputStyle} disabled={isRunning} />
              </div>
              <div>
                <label style={labelStyle}>Password</label>
                <input type="password" value={password} onChange={e => setPassword(e.target.value)} style={inputStyle} disabled={isRunning} />
              </div>
            </div>
          )}

          {drivers.length > 1 && (
            <div style={{marginBottom: 12}}>
              <label style={labelStyle}>ODBC Driver</label>
              <select value={odbcDriver} onChange={e => setOdbcDriver(e.target.value)} style={inputStyle} disabled={isRunning}>
                {drivers.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
            </div>
          )}

          <div style={{marginBottom: 12}}>
            <label style={{...labelStyle, display: "flex", alignItems: "center", gap: 8, cursor: "pointer"}}>
              <input type="checkbox" checked={applyCci} onChange={e => setApplyCci(e.target.checked)} disabled={isRunning} />
              Apply Columnstore Indexes (CCI)
            </label>
            <div style={{fontSize: 11, color: "var(--muted)", marginTop: 2, paddingLeft: 22}}>Improves analytics query performance. Adds 30-60s to import time.</div>
          </div>
        </div>

        {/* Right column: Dataset */}
        <div>
          <div style={{fontSize: 10.5, fontWeight: 700, color: "var(--muted)", letterSpacing: ".09em", textTransform: "uppercase", marginBottom: 12}}>Dataset</div>

          {datasetsLoading ? (
            <div style={{padding: 20, textAlign: "center", color: "var(--muted)", fontSize: 12}}>Loading datasets...</div>
          ) : datasets.length === 0 ? (
            <div style={{padding: 20, textAlign: "center", color: "var(--muted)", fontSize: 12}}>No CSV datasets found. Generate data with CSV format first.</div>
          ) : (
            <div style={{display: "flex", flexDirection: "column", gap: 4, maxHeight: 320, overflowY: "auto"}}>
              {datasets.map(ds => {
                const isSel = selectedDataset === ds.name;
                return (
                  <button key={ds.name} onClick={() => !isRunning && setSelectedDataset(ds.name)} style={{
                    display: "flex", alignItems: "center", gap: 10, width: "100%", padding: "9px 12px", borderRadius: 8,
                    cursor: isRunning ? "default" : "pointer",
                    border: `1px solid ${isSel ? "var(--accent)" : "var(--border)"}`,
                    background: isSel ? "var(--glow)" : "var(--surface)", fontFamily: "var(--sans)",
                    transition: "all .12s", textAlign: "left", opacity: isRunning ? .6 : 1,
                  }} onMouseOver={e => { if (!isSel && !isRunning) e.currentTarget.style.borderColor = "var(--accent)"; }}
                     onMouseOut={e => { if (!isSel) e.currentTarget.style.borderColor = "var(--border)"; }}>
                    <span style={{fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)", flexShrink: 0}}>{ds.date}</span>
                    <span style={{fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)", flexShrink: 0}}>{ds.time}</span>
                    <span style={{flex: 1, fontSize: 12, fontWeight: isSel ? 600 : 400, color: isSel ? "var(--accent)" : "var(--text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap"}}>{ds.description}</span>
                    <span style={{fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)"}}>{ds.table_count} tbls</span>
                    <span style={{fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)"}}>{ds.size_mb} MB</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Action buttons */}
      <div style={{marginBottom: 16}}>
        {isRunning ? (
          <div style={{display: "flex", gap: 10}}>
            <button disabled style={{flex: 1, padding: "12px 24px", borderRadius: 10, border: "none", fontSize: 14, fontWeight: 700, fontFamily: "var(--sans)", background: "var(--accent)", color: "#fff", opacity: .7, cursor: "default"}}>
              Importing... {elapsed < 60 ? elapsed.toFixed(1) + "s" : Math.floor(elapsed / 60) + "m " + Math.floor(elapsed % 60) + "s"}
            </button>
            <button onClick={cancelImport} style={{padding: "12px 20px", borderRadius: 10, border: "1px solid var(--err)", background: "var(--errBg)", color: "var(--err)", fontWeight: 600, fontSize: 13, cursor: "pointer", fontFamily: "var(--sans)"}}>Cancel</button>
          </div>
        ) : (
          <button onClick={startImport} disabled={!canStart || driverAvailable === false} style={{
            width: "100%", padding: "12px 24px", borderRadius: 10, border: "none", fontSize: 14, fontWeight: 700,
            cursor: canStart && driverAvailable !== false ? "pointer" : "not-allowed", fontFamily: "var(--sans)", transition: "all .15s",
            background: canStart && driverAvailable !== false ? "var(--accent)" : "var(--alt)",
            color: canStart && driverAvailable !== false ? "#fff" : "var(--muted)",
            boxShadow: canStart && driverAvailable !== false ? "0 4px 16px rgba(79,91,213,.2)" : "none",
          }}>Import to SQL Server</button>
        )}
      </div>

      {/* Status badge */}
      {jobStatus && !isRunning && (
        <div style={{
          padding: "10px 14px", marginBottom: 12, borderRadius: 8, fontSize: 12.5, display: "flex", alignItems: "center", gap: 8,
          background: jobStatus === "done" ? "rgba(34,197,94,.06)" : "var(--errBg)",
          border: `1px solid ${jobStatus === "done" ? "rgba(34,197,94,.12)" : "rgba(239,68,68,.15)"}`,
          color: jobStatus === "done" ? "var(--ok)" : "var(--err)",
        }}>
          <span style={{fontWeight: 700}}>{jobStatus === "done" ? "\u2713" : "\u2717"}</span>
          <span>
            {jobStatus === "done" ? "Import completed successfully" : jobStatus === "cancelled" ? "Import cancelled" : "Import failed"}
            {" \u2014 "}{elapsed < 60 ? elapsed.toFixed(1) + "s" : Math.floor(elapsed / 60) + "m " + Math.floor(elapsed % 60) + "s"}
          </span>
        </div>
      )}

      {/* Log viewer */}
      {logs.length > 0 && (
        <div style={{border: "1px solid var(--border)", borderRadius: 8, overflow: "hidden"}}>
          <div style={{padding: "8px 12px", background: "var(--alt)", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between"}}>
            <span style={{fontSize: 11, fontWeight: 600, color: "var(--dim)"}}>Import Log</span>
            <span style={{fontSize: 11, color: "var(--muted)", fontFamily: "var(--mono)"}}>{logs.length} lines</span>
          </div>
          <div style={{maxHeight: 360, overflowY: "auto", padding: "8px 12px", background: "var(--surface)", fontFamily: "var(--mono)", fontSize: 11.5, lineHeight: 1.6}}>
            {logs.map((line, idx) => {
              const isErr = /FAIL|ERROR/i.test(line);
              const isDone = /DONE/i.test(line);
              const isWarn = /WARN/i.test(line);
              return (
                <div key={idx} style={{color: isErr ? "var(--err)" : isDone ? "var(--ok)" : isWarn ? "var(--warn)" : "var(--text)", whiteSpace: "pre-wrap", wordBreak: "break-all"}}>{line}</div>
              );
            })}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}
