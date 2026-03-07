/* ═══ app.jsx — main application ═══ */

/* Client-side validation */
function validate(c){
  const e=[],w=[];
  if(c.endDate<c.startDate) e.push("End date must be after start date.");
  if(c.salesRows<=0) e.push("Total rows must be > 0.");
  if(c.maxPrice<=c.minPrice) e.push("Max unit price must exceed min unit price.");
  if(c.returnsEnabled&&(c.returnRate<0||c.returnRate>1)) e.push("Return rate must be 0\u20131.");
  const rS=c.pctIndia+c.pctUs+c.pctEu+c.pctAsia;
  if(rS<=0) w.push("Customer regional % sum to 0.");
  if(c.chunkSize>c.salesRows) w.push("Chunk size exceeds total rows.");
  if(c.format==="csv"&&c.salesRows>5000000) w.push("Large CSV outputs can be slow. Consider parquet.");
  if(c.skipOrderCols) w.push("Order columns skipped \u2014 Returns will not be generated.");
  if(c.customers>c.salesRows&&c.salesRows>0) w.push(`Customers (${c.customers.toLocaleString()}) exceed sales rows.`);
  if(c.products>c.salesRows&&c.salesRows>0) w.push(`Products (${c.products.toLocaleString()}) exceed sales rows.`);
  if(c.customers>0&&c.salesRows/c.customers<1.5) w.push(`Low rows-per-customer ratio (${(c.salesRows/c.customers).toFixed(1)}).`);
  const gS=Object.values(c.geoWeights).reduce((a,b)=>a+b,0);
  if(Math.abs(gS-1)>.05&&gS>0) w.push(`Geography weights sum to ${(gS*100).toFixed(0)}% (expected ~100%).`);
  if(c.returnsEnabled&&c.returnMaxDays<=c.returnMinDays) w.push("Return max days should exceed min days.");
  return{errors:e,warnings:w};
}

/* Debounced sync */
function useDebounce(fn,ms){
  const t=useRef(null);
  return useCallback((...a)=>{clearTimeout(t.current);t.current=setTimeout(()=>fn(...a),ms);},[fn,ms]);
}

/* ═══════════════════════════════════════════════════════════════════
   App
   ═══════════════════════════════════════════════════════════════════ */
function App(){
  const[cfg,setCfg]=useState(null);
  const[presets,setPresets]=useState({});
  const[presetBucket,setPresetBucket]=useState("");
  const[selectedPreset,setSelectedPreset]=useState(null);
  const[toast,setToast]=useState(null);
  const[logs,setLogs]=useState([]);
  const[isRunning,setIsRunning]=useState(false);
  const[elapsed,setElapsed]=useState(0);
  const timerRef=useRef(null);

  /* Models YAML state */
  const[modelsYaml,setModelsYaml]=useState("");
  const[modelsOrig,setModelsOrig]=useState("");
  const[modelsDisk,setModelsDisk]=useState("");
  const[modelsErr,setModelsErr]=useState(null);
  const modelsDirty=modelsYaml!==modelsOrig;
  const modelsApplied=modelsOrig!==modelsDisk;

  /* Config YAML state */
  const[cfgYaml,setCfgYaml]=useState("");
  const[cfgYamlOrig,setCfgYamlOrig]=useState("");
  const[cfgYamlDisk,setCfgYamlDisk]=useState("");
  const[cfgYamlCurrent,setCfgYamlCurrent]=useState("");
  const[cfgYamlErr,setCfgYamlErr]=useState(null);
  const cfgYamlDirty=cfgYaml!==cfgYamlOrig;
  const cfgYamlApplied=cfgYamlCurrent!==cfgYamlDisk;

  /* Page navigation */
  const[page,setPage_]=useState("main");
  const setPage=useCallback((p)=>{
    document.activeElement?.blur();
    const scrollY=window.scrollY;
    setPage_(p);
    if(p==="config")loadCfgYaml();
    if(p==="models")fetch(API+"/models").then(r=>r.text()).then(t=>{setModelsYaml(t);setModelsOrig(t);}).catch(()=>{});
    const restore=()=>window.scrollTo(0,scrollY);
    restore();
    requestAnimationFrame(()=>{restore();requestAnimationFrame(restore);});
  },[]);

  /* ─── Initial load ─── */
  useEffect(()=>{
    fetch(API+"/config").then(r=>r.json()).then(d=>{
      d.geoWeights=d.geoWeights||{};
      d.regenAll=false;d.regenDims={};d.autoWorkers=!d.workers;
      setCfg(d);
    }).catch(()=>{
      setCfg({format:"parquet",salesOutput:"sales",skipOrderCols:false,compression:"snappy",rowGroupSize:2000000,mergeParquet:true,partitionEnabled:true,startDate:"2020-01-01",endDate:"2025-12-31",fiscalMonthOffset:0,includeCalendar:true,includeIso:false,includeFiscal:true,includeWeeklyFiscal:false,wfFirstDay:0,wfWeeklyType:"Last",wfQuarterType:"445",salesRows:103285,chunkSize:1000000,autoWorkers:false,workers:8,customers:48837,stores:10,products:2581,promotions:20,pctIndia:10,pctUs:51,pctEu:39,pctAsia:0,pctOrg:1,customerActiveRatio:.98,profile:"steady",firstYearPct:.27,valueScale:1,minPrice:10,maxPrice:3000,productActiveRatio:.98,geoWeights:{"United States":.35,India:.2,"United Kingdom":.1,Germany:.1,France:.1,Australia:.07,Canada:.08},returnsEnabled:true,returnRate:.03,returnMinDays:1,returnMaxDays:60,budgetEnabled:true,inventoryEnabled:true,regenAll:false,regenDims:{}});
    });
    fetch(API+"/presets").then(r=>r.json()).then(d=>{setPresets(d);setPresetBucket(Object.keys(d)[0]||"");}).catch(()=>{});
    fetch(API+"/models").then(r=>r.text()).then(t=>{setModelsYaml(t);setModelsOrig(t);setModelsDisk(t);}).catch(()=>{});
  },[]);

  /* ─── Debounced config sync ─── */
  const syncToBackend=useDebounce((vals)=>{
    fetch(API+"/config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({values:vals})}).catch(()=>{});
  },400);

  const s=useCallback((k,v)=>{
    setCfg(p=>{const n={...p,[k]:v};syncToBackend(n);return n;});
  },[syncToBackend]);

  const reloadFormFromServer=useCallback(()=>{
    fetch(API+"/config").then(r=>r.json()).then(d=>{
      d.geoWeights=d.geoWeights||{};d.regenAll=cfg?.regenAll||false;d.regenDims=cfg?.regenDims||{};d.autoWorkers=!d.workers;
      setCfg(d);
    }).catch(()=>{});
  },[]);

  const loadCfgYaml=useCallback(()=>{
    Promise.all([
      fetch(API+"/config/yaml").then(r=>r.text()),
      fetch(API+"/config/yaml/disk").then(r=>r.text()),
    ]).then(([current,disk])=>{
      setCfgYamlCurrent(current);
      setCfgYamlDisk(disk);
      const initial=(disk&&disk.trim().length>0)?disk:current;
      setCfgYaml(initial);setCfgYamlOrig(initial);setCfgYamlErr(null);
    }).catch(()=>{});
  },[]);

  const flash=m=>{setToast(m);setTimeout(()=>setToast(null),2400);};
  const setGeo=(c,v)=>setCfg(p=>{const n={...p,geoWeights:{...p.geoWeights,[c]:Math.max(0,Math.min(1,v))}};syncToBackend(n);return n;});

  const applyPreset=(name)=>{
    fetch(API+"/presets/apply",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name})})
      .then(()=>fetch(API+"/config").then(r=>r.json()).then(d=>{d.geoWeights=d.geoWeights||{};d.regenAll=cfg.regenAll;d.regenDims=cfg.regenDims;d.autoWorkers=!d.workers;setCfg(d);}))
      .catch(()=>{});
    setSelectedPreset(name);
    flash(`Applied: ${name}`);
  };

  /* ─── Generate ─── */
  const runGenerate=()=>{
    if(errors.length>0||isRunning)return;
    const regenDims=cfg.regenAll?DIMS:Object.keys(cfg.regenDims).filter(k=>cfg.regenDims[k]);
    fetch(API+"/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({regen_dimensions:regenDims.length?regenDims:null})})
      .then(r=>r.json()).then(()=>{
        setLogs([]);setIsRunning(true);setElapsed(0);
        const start=Date.now();
        timerRef.current=setInterval(()=>setElapsed((Date.now()-start)/1000),100);
        const es=new EventSource(API+"/generate/stream");
        es.onmessage=(ev)=>{
          const d=JSON.parse(ev.data);
          if(d.type==="log")setLogs(p=>[...p,d.line]);
          if(d.type==="status")setElapsed(d.elapsed);
          if(d.type==="end"){clearInterval(timerRef.current);setIsRunning(false);setElapsed(d.elapsed);es.close();}
          if(d.type==="idle"){es.close();}
        };
        es.onerror=()=>{clearInterval(timerRef.current);setIsRunning(false);es.close();};
      }).catch(e=>{setLogs(["Failed to start pipeline: "+e.message]);});
  };

  const cancelGenerate=()=>{
    fetch(API+"/generate/cancel",{method:"POST"}).then(()=>{clearInterval(timerRef.current);setIsRunning(false);}).catch(()=>{});
  };

  /* ─── Models YAML ─── */
  const saveModels=()=>{
    setModelsErr(null);
    fetch(API+"/models",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({yaml_text:modelsYaml})})
      .then(r=>{if(!r.ok)return r.json().then(d=>{throw new Error(d.detail||"Save failed")});return r.json();})
      .then(d=>{
        setModelsOrig(modelsYaml);setModelsErr(null);flash("Models saved (in memory)");
      })
      .catch(e=>setModelsErr(e.message));
  };

  const resetModels=()=>{
    fetch(API+"/models/reset",{method:"POST"})
      .then(r=>r.json())
      .then(d=>{
        fetch(API+"/models").then(r=>r.text()).then(t=>{setModelsYaml(t);setModelsOrig(t);setModelsDisk(t);setModelsErr(null);flash("Models reset to disk version");});
      }).catch(()=>{});
  };

  /* ─── Config YAML ─── */
  const saveCfgYaml=()=>{
    setCfgYamlErr(null);
    fetch(API+"/config/yaml",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({yaml_text:cfgYaml})})
      .then(r=>{if(!r.ok)return r.json().then(d=>{throw new Error(d.detail||"Save failed")});return r.json();})
      .then(()=>{setCfgYamlOrig(cfgYaml);setCfgYamlCurrent(cfgYaml);setCfgYamlErr(null);flash("Config applied (in memory)");reloadFormFromServer();})
      .catch(e=>setCfgYamlErr(e.message));
  };

  const resetCfgYaml=()=>{
    fetch(API+"/config/yaml/reset",{method:"POST"})
      .then(r=>r.json())
      .then(()=>{loadCfgYaml();reloadFormFromServer();flash("Config reset to disk version");})
      .catch(()=>{});
  };

  const refreshCfgYamlFromUI=()=>{
    fetch(API+"/config/yaml").then(r=>r.text()).then(t=>{setCfgYamlCurrent(t);setCfgYaml(t);setCfgYamlOrig(t);setCfgYamlErr(null);flash("Loaded current UI settings");}).catch(()=>{});
  };

  /* ─── Render ─── */
  if(!cfg)return <div style={{padding:40,textAlign:"center",color:"var(--muted)"}}>Loading configuration...</div>;
  const{errors,warnings}=validate(cfg);
  const showPq=cfg.format==="parquet"||cfg.format==="deltaparquet";
  const showDelta=cfg.format==="deltaparquet";

  return(
    <div style={{display:"flex",minHeight:"100vh"}}>
      {toast&&<div style={{position:"fixed",top:16,right:16,zIndex:999,background:"var(--accent)",color:"#fff",padding:"9px 20px",borderRadius:8,fontSize:13,fontWeight:600,boxShadow:"0 6px 24px rgba(79,91,213,.25)",animation:"slideIn .28s ease"}}>{toast}</div>}

      {/* ═══ SIDEBAR ═══ */}
      <div style={{width:360,flexShrink:0,background:"var(--surface)",borderRight:"1px solid var(--border)",padding:20,overflowY:"auto"}}>
        <div style={{fontSize:10.5,fontWeight:700,color:"var(--muted)",letterSpacing:".09em",textTransform:"uppercase",marginBottom:10}}>Presets</div>
        {Object.keys(presets).length>0?(<>
          <Sel value={presetBucket} onChange={setPresetBucket} options={Object.keys(presets)} />
          <div style={{marginTop:8,display:"flex",flexDirection:"column",gap:4}}>{Object.entries(presets[presetBucket]||{}).map(([name,p])=>{
            const sel=selectedPreset===name;
            const c=p||{};const cust=c.customers||c.total_customers;const prod=c.products||c.num_products;const rows=c.sales_rows||c.total_rows;
            return(
              <button key={name} onClick={()=>applyPreset(name)} style={{display:"flex",flexDirection:"column",alignItems:"flex-start",width:"100%",padding:"9px 12px",borderRadius:8,cursor:"pointer",border:`1px solid ${sel?"var(--accent)":"var(--border)"}`,background:sel?"var(--glow)":"var(--surface)",fontFamily:"var(--sans)",transition:"all .12s",gap:4}}>
                <span style={{fontSize:12.5,fontWeight:sel?600:400,color:sel?"var(--accent)":"var(--text)",textAlign:"left",lineHeight:1.35}}>{name}</span>
                {(cust||prod||rows)&&<span style={{display:"flex",gap:5}}>
                  {rows&&<span style={{fontSize:10,padding:"1px 6px",borderRadius:4,background:sel?"rgba(79,91,213,.12)":"var(--alt)",color:sel?"var(--accent)":"var(--muted)",fontFamily:"var(--mono)",fontWeight:500}}>{rows>=1000000?(rows/1000000).toFixed(1)+"M":(rows/1000).toFixed(rows%1000?1:0)+"K"} rows</span>}
                  {cust&&<span style={{fontSize:10,padding:"1px 6px",borderRadius:4,background:sel?"rgba(79,91,213,.12)":"var(--alt)",color:sel?"var(--accent)":"var(--muted)",fontFamily:"var(--mono)",fontWeight:500}}>{(cust/1000).toFixed(cust%1000?1:0)}K cust</span>}
                  {prod&&<span style={{fontSize:10,padding:"1px 6px",borderRadius:4,background:sel?"rgba(79,91,213,.12)":"var(--alt)",color:sel?"var(--accent)":"var(--muted)",fontFamily:"var(--mono)",fontWeight:500}}>{(prod/1000).toFixed(prod%1000?1:0)}K prod</span>}
                </span>}
              </button>
            );
          })}</div>
        </>):(<div style={{fontSize:12,color:"var(--muted)"}}>No presets loaded</div>)}

        <div style={{borderTop:"1px solid var(--border)",margin:"20px 0"}} />
        <div style={{fontSize:10.5,fontWeight:700,color:"var(--muted)",letterSpacing:".09em",textTransform:"uppercase",marginBottom:10}}>Models</div>
        <button onClick={()=>setPage(page==="models"?"main":"models")} style={{width:"100%",padding:"9px 12px",borderRadius:8,border:`1px solid ${page==="models"?"var(--accent)":"var(--border)"}`,background:page==="models"?"var(--glow)":"var(--surface)",color:page==="models"?"var(--accent)":"var(--dim)",fontSize:12.5,fontWeight:page==="models"?600:500,cursor:"pointer",fontFamily:"var(--sans)",transition:"all .15s",display:"flex",alignItems:"center",justifyContent:"space-between"}}>
          <span>Models Configuration</span>
          {(modelsDirty||modelsApplied)&&<span style={{width:7,height:7,borderRadius:"50%",background:modelsDirty?"var(--warn)":"var(--accent)",flexShrink:0}} />}
        </button>
        {page==="models"&&<div style={{fontSize:11,color:"var(--muted)",marginTop:6}}>Editing models.yaml in memory</div>}

        <div style={{borderTop:"1px solid var(--border)",margin:"20px 0"}} />
        <div style={{fontSize:10.5,fontWeight:700,color:"var(--muted)",letterSpacing:".09em",textTransform:"uppercase",marginBottom:10}}>Config</div>
        <button onClick={()=>setPage(page==="config"?"main":"config")} style={{width:"100%",padding:"9px 12px",borderRadius:8,border:`1px solid ${page==="config"?"var(--accent)":"var(--border)"}`,background:page==="config"?"var(--glow)":"var(--surface)",color:page==="config"?"var(--accent)":"var(--dim)",fontSize:12.5,fontWeight:page==="config"?600:500,cursor:"pointer",fontFamily:"var(--sans)",transition:"all .15s",display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:6}}>
          <span>Config YAML</span>
          {(cfgYamlDirty||cfgYamlApplied)&&<span style={{width:7,height:7,borderRadius:"50%",background:cfgYamlDirty?"var(--warn)":"var(--accent)",flexShrink:0}} />}
        </button>
        {page==="config"&&<div style={{fontSize:11,color:"var(--muted)",marginTop:6,marginBottom:6}}>Editing config.yaml in memory</div>}
        <button onClick={()=>{
          Promise.all([
            fetch(API+"/config/yaml").then(r=>r.text()),
            fetch(API+"/models").then(r=>r.text()),
          ]).then(([cfgText,modelsText])=>{
            const dl=(content,name,type)=>{const b=new Blob([content],{type});const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=name;a.click();URL.revokeObjectURL(a.href);};
            dl(cfgText,"config.yaml","text/yaml");
            setTimeout(()=>dl(modelsText,"models.yaml","text/yaml"),100);
          }).catch(()=>{});
        }} style={{width:"100%",padding:"8px 12px",borderRadius:8,border:"1px solid var(--border)",background:"var(--surface)",color:"var(--dim)",fontSize:12,cursor:"pointer",fontFamily:"var(--sans)"}}>Download configs</button>
      </div>

      {/* ═══ MAIN ═══ */}
      <div style={{flex:1,padding:"28px 40px",maxWidth:1120,overflowY:"auto"}}>

        {page==="models"?(
        /* ═══ MODELS EDITOR PAGE ═══ */
        <div>
          <button onClick={()=>setPage("main")} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:7,border:"1px solid var(--border)",background:"var(--surface)",color:"var(--dim)",fontSize:12.5,cursor:"pointer",fontFamily:"var(--sans)",marginBottom:18,transition:"all .12s"}} onMouseOver={e=>e.currentTarget.style.borderColor="var(--accent)"} onMouseOut={e=>e.currentTarget.style.borderColor="var(--border)"}>
            {"\u2190"} Back to configuration
          </button>
          <h1 style={{fontSize:22,fontWeight:700,letterSpacing:"-.02em",marginBottom:4}}>Models Configuration</h1>
          <p style={{fontSize:13,color:"var(--dim)",marginTop:3,marginBottom:18}}>
            Edit the full models configuration. Changes are held in memory only {"\u2014"} the file on disk is never modified.
          </p>
          <YamlEditor value={modelsYaml} onChange={setModelsYaml} filename="models.yaml" dirty={modelsDirty} applied={modelsApplied} error={modelsErr} onApply={saveModels} onReset={resetModels} />
        </div>

        ):page==="config"?(
        /* ═══ CONFIG YAML EDITOR PAGE ═══ */
        <div>
          <button onClick={()=>setPage("main")} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:7,border:"1px solid var(--border)",background:"var(--surface)",color:"var(--dim)",fontSize:12.5,cursor:"pointer",fontFamily:"var(--sans)",marginBottom:18,transition:"all .12s"}} onMouseOver={e=>e.currentTarget.style.borderColor="var(--accent)"} onMouseOut={e=>e.currentTarget.style.borderColor="var(--border)"}>
            {"\u2190"} Back to configuration
          </button>
          <h1 style={{fontSize:22,fontWeight:700,letterSpacing:"-.02em",marginBottom:4}}>Config YAML</h1>
          <p style={{fontSize:13,color:"var(--dim)",marginTop:3,marginBottom:18}}>
            Edit the full config below. Applying here will override any preset or form settings. The file on disk is never modified.
          </p>
          <YamlEditor value={cfgYaml} onChange={setCfgYaml} filename="config.yaml" dirty={cfgYamlDirty} applied={cfgYamlApplied} error={cfgYamlErr} onApply={saveCfgYaml} onReset={resetCfgYaml} onRefresh={refreshCfgYamlFromUI} />
        </div>

        ):(
        /* ═══ MAIN CONFIG PAGE ═══ */
        <div>
        <h1 style={{fontSize:22,fontWeight:700,letterSpacing:"-.02em"}}>Retail Data Generator</h1>
        <p style={{fontSize:13,color:"var(--dim)",marginTop:3,marginBottom:22}}>Configure and generate large, realistic retail datasets</p>

        {/* 1 OUTPUT */}
        <Section num="1" title="Output">
          <R2>
            <F label="Output format"><Sel value={cfg.format} onChange={v=>s("format",v)} options={["parquet","csv","deltaparquet"]} labels={["Parquet","CSV","Delta (Parquet)"]} /></F>
            <F label="Sales output"><Sel value={cfg.salesOutput} onChange={v=>s("salesOutput",v)} options={["sales","sales_order","both"]} labels={["Sales (flat)","Order Header + Detail","Both"]} /></F>
          </R2>
          <div style={{marginTop:12}}><Check checked={cfg.skipOrderCols} onChange={v=>s("skipOrderCols",v)} label="Skip order columns" /></div>
          {showPq&&<Box title="Parquet options"><R3>
            <F label="Compression"><Sel value={cfg.compression} onChange={v=>s("compression",v)} options={["snappy","zstd","gzip","none"]} /></F>
            <F label="Row group size"><N value={cfg.rowGroupSize} onChange={v=>s("rowGroupSize",v)} min={100000} step={500000} /></F>
            <F label=" "><div style={{display:"flex",flexDirection:"column",gap:8,paddingTop:3}}><Check checked={cfg.mergeParquet} onChange={v=>s("mergeParquet",v)} label="Merge parquet chunks" />{showDelta&&<Check checked={cfg.partitionEnabled} onChange={v=>s("partitionEnabled",v)} label="Partition by Year/Month" />}</div></F>
          </R3></Box>}
        </Section>

        {/* 2 DATES */}
        <Section num="2" title="Dates">
          <R2>
            <F label="Start date"><input type="date" style={iS} value={cfg.startDate} onChange={e=>s("startDate",e.target.value)} /></F>
            <F label="End date"><input type="date" style={iS} value={cfg.endDate} onChange={e=>s("endDate",e.target.value)} /></F>
          </R2>
          <Box title="Calendar modes">
            <R4>
              <div style={{paddingTop:2}}><Check checked={true} onChange={()=>{}} label="Calendar" disabled /><div style={{fontSize:10,color:"var(--muted)",marginTop:2,marginLeft:24}}>Always on</div></div>
              <Check checked={cfg.includeIso} onChange={v=>s("includeIso",v)} label="ISO weeks" />
              <Check checked={cfg.includeFiscal} onChange={v=>s("includeFiscal",v)} label="Fiscal" />
              <Check checked={cfg.includeWeeklyFiscal} onChange={v=>s("includeWeeklyFiscal",v)} label="Weekly fiscal (4-4-5)" />
            </R4>
            {(cfg.includeFiscal||cfg.includeWeeklyFiscal)&&<div style={{marginTop:14}}><R3>
              <F label="Fiscal year starts"><Sel value={String(cfg.fiscalMonthOffset)} onChange={v=>s("fiscalMonthOffset",parseInt(v))} options={MONTHS.map((_,i)=>String(i))} labels={MONTHS} /></F>
              {cfg.includeWeeklyFiscal&&<><F label="First day of week"><Sel value={String(cfg.wfFirstDay)} onChange={v=>s("wfFirstDay",parseInt(v))} options={DAYS.map((_,i)=>String(i))} labels={DAYS} /></F><F label="Quarter pattern"><Sel value={cfg.wfQuarterType} onChange={v=>s("wfQuarterType",v)} options={["445","454","544"]} /></F></>}
            </R3></div>}
          </Box>
        </Section>

        {/* 3 VOLUME */}
        <Section num="3" title="Volume">
          <F label="Sales rows" help="Total rows to generate in the Sales fact table."><N value={cfg.salesRows} onChange={v=>s("salesRows",v)} min={1} step={10000} /></F>
          <R2>
            <F label="Chunk size"><N value={cfg.chunkSize} onChange={v=>s("chunkSize",v)} min={10000} step={100000} /></F>
            <F label="Workers"><div style={{display:"flex",alignItems:"center",gap:10}}><Check checked={cfg.autoWorkers} onChange={v=>{s("autoWorkers",v);if(v)s("workers",0);}} label="Auto" />{!cfg.autoWorkers&&<N value={cfg.workers} onChange={v=>s("workers",v)} min={1} max={32} style={{width:100}} />}</div></F>
          </R2>
        </Section>

        {/* 4 DIMENSIONS */}
        <Section num="4" title="Dimension Sizes">
          <R4>
            <F label="Customers"><N value={cfg.customers} onChange={v=>s("customers",v)} min={1} step={1000} /></F>
            <F label="Products"><N value={cfg.products} onChange={v=>s("products",v)} min={1} step={500} /></F>
            <F label="Stores"><N value={cfg.stores} onChange={v=>s("stores",v)} min={1} step={10} /></F>
            <F label="Promotions"><N value={cfg.promotions} onChange={v=>s("promotions",v)} min={0} step={5} /></F>
          </R4>
        </Section>

        {/* 5 CUSTOMERS */}
        <Section num="5" title="Customers" defaultOpen={false}>
          <Box title="Regional mix (% of customers)">
            <R4>
              <F label="India %"><N value={cfg.pctIndia} onChange={v=>s("pctIndia",v)} min={0} max={100} /></F>
              <F label="US %"><N value={cfg.pctUs} onChange={v=>s("pctUs",v)} min={0} max={100} /></F>
              <F label="EU %"><N value={cfg.pctEu} onChange={v=>s("pctEu",v)} min={0} max={100} /></F>
              <F label="Asia %"><N value={cfg.pctAsia} onChange={v=>s("pctAsia",v)} min={0} max={100} /></F>
            </R4>
            <div style={{fontSize:11,color:"var(--muted)",marginTop:6}}>Sum: {cfg.pctIndia+cfg.pctUs+cfg.pctEu+cfg.pctAsia} (auto-normalized)</div>
            <Sld label="Organization %" value={cfg.pctOrg} min={0} max={100} step={1} onChange={v=>s("pctOrg",v)} fmt={v=>`${v}%`} />
          </Box>
          <Sld label="Active ratio" value={cfg.customerActiveRatio} min={.1} max={1} step={.01} onChange={v=>s("customerActiveRatio",v)} />
          <Box title="Behavior Profile">
            <R2>
              <F label="Profile" help="Controls acquisition curve, churn, seasonality, and demand shape."><Sel value={cfg.profile} onChange={v=>s("profile",v)} options={["gradual","steady","aggressive","instant"]} labels={["Gradual (S-curve ramp)","Steady (mature business)","Aggressive (fast growth)","Instant (all customers day 1)"]} /></F>
              <F label="First year %" help="% of customers that exist in year 1. Rest are acquired over time."><N value={cfg.firstYearPct} onChange={v=>s("firstYearPct",v)} min={0.05} max={1} step={0.01} /></F>
            </R2>
          </Box>
        </Section>

        {/* 6 PRODUCTS */}
        <Section num="6" title="Products" defaultOpen={false}>
          <Box title="Pricing">
            <R3>
              <F label="Value scale" help="Multiplier on base product prices."><N value={cfg.valueScale} onChange={v=>s("valueScale",v)} min={.01} max={10} step={.05} /></F>
              <F label="Min unit price"><N value={cfg.minPrice} onChange={v=>s("minPrice",v)} min={0} step={10} /></F>
              <F label="Max unit price"><N value={cfg.maxPrice} onChange={v=>s("maxPrice",v)} min={1} step={50} /></F>
            </R3>
            {cfg.maxPrice>cfg.minPrice&&<div style={{fontSize:11,color:"var(--muted)",marginTop:8}}>Scaled range: ~{((cfg.minPrice||0)*(cfg.valueScale||1)).toLocaleString()} {"\u2192"} {((cfg.maxPrice||0)*(cfg.valueScale||1)).toLocaleString()}</div>}
          </Box>
          <Sld label="Active ratio" value={cfg.productActiveRatio} min={.1} max={1} step={.01} onChange={v=>s("productActiveRatio",v)} />
        </Section>

        {/* 7 GEOGRAPHY */}
        <Section num="7" title="Geography" defaultOpen={false}>
          <F label="Country weights" help="Relative distribution of geography rows. Auto-normalized.">
            <div style={{marginTop:8}}>{Object.entries(cfg.geoWeights).map(([country,w])=>(
              <div key={country} style={{display:"flex",alignItems:"center",gap:10,marginBottom:6}}>
                <span style={{fontSize:13,width:140,color:"var(--text)"}}>{country}</span>
                <input type="range" min={0} max={.5} step={.01} value={w} onChange={e=>setGeo(country,parseFloat(e.target.value))} style={{flex:1,accentColor:"var(--accent)"}} />
                <span style={{fontSize:12,fontFamily:"var(--mono)",color:"var(--accent)",width:42,textAlign:"right"}}>{(w*100).toFixed(0)}%</span>
              </div>
            ))}</div>
          </F>
        </Section>

        {/* 8 RETURNS */}
        <Section num="8" title="Returns" defaultOpen={false} badge={cfg.returnsEnabled?{v:"success",t:"ON"}:{v:"default",t:"OFF"}}>
          <div style={{marginTop:10}}><Check checked={cfg.returnsEnabled} onChange={v=>s("returnsEnabled",v)} label="Enable returns generation" /></div>
          {cfg.returnsEnabled&&<R3>
            <F label="Return rate" help="Fraction of sales rows returned."><N value={cfg.returnRate} onChange={v=>s("returnRate",v)} min={0} max={1} step={.005} /></F>
            <F label="Min days after sale"><N value={cfg.returnMinDays} onChange={v=>s("returnMinDays",v)} min={1} step={1} /></F>
            <F label="Max days after sale"><N value={cfg.returnMaxDays} onChange={v=>s("returnMaxDays",v)} min={1} step={5} /></F>
          </R3>}
        </Section>

        {/* 9 BUDGET & INVENTORY */}
        <Section num="9" title="Budget & Inventory" defaultOpen={false}>
          <div style={{display:"flex",flexDirection:"column",gap:10,marginTop:10}}>
            <Check checked={cfg.budgetEnabled} onChange={v=>s("budgetEnabled",v)} label="Generate Budget fact table" />
            <Check checked={cfg.inventoryEnabled} onChange={v=>s("inventoryEnabled",v)} label="Generate Inventory Snapshot fact table" />
          </div>
        </Section>

        {/* 10 REGENERATE */}
        <Section num="10" title="Regenerate Dimensions" defaultOpen={false}>
          <div style={{marginTop:10}}><Check checked={cfg.regenAll} onChange={v=>s("regenAll",v)} label="Regenerate all dimensions" /></div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8,marginTop:12}}>{DIMS.map(d=><Check key={d} checked={cfg.regenAll||!!cfg.regenDims[d]} onChange={v=>s("regenDims",{...cfg.regenDims,[d]:v})} label={d.replace("_"," ")} />)}</div>
        </Section>

        {/* 11 VALIDATION */}
        <Section num="11" title="Validation" badge={errors.length>0?{v:"error",t:`${errors.length} error${errors.length>1?"s":""}`}:warnings.length>0?{v:"warning",t:`${warnings.length} warning${warnings.length>1?"s":""}`}:{v:"success",t:"Valid"}}>
          <div style={{marginTop:8}}>
            {errors.length===0&&warnings.length===0&&<div style={{display:"flex",alignItems:"center",gap:8}}><Badge variant="success">Valid</Badge><span style={{fontSize:13,color:"var(--dim)"}}>Configuration looks good.</span></div>}
            {errors.map((e,i)=><div key={`e${i}`} style={{padding:"9px 13px",borderRadius:8,marginTop:6,fontSize:12.5,display:"flex",alignItems:"center",gap:7,background:"var(--errBg)",color:"var(--err)",border:"1px solid rgba(220,38,38,.12)"}}><span style={{fontWeight:700}}>✕</span> {e}</div>)}
            {warnings.map((w,i)=><div key={`w${i}`} style={{padding:"9px 13px",borderRadius:8,marginTop:6,fontSize:12.5,display:"flex",alignItems:"center",gap:7,background:"var(--warnBg)",color:"var(--warn)",border:"1px solid rgba(202,138,4,.1)"}}><span style={{fontWeight:700}}>⚠</span> {w}</div>)}
          </div>
        </Section>

        {/* 12 GENERATE */}
        <Section num="12" title="Generate">
          <div style={{marginTop:10,padding:"12px 16px",background:"var(--alt)",borderRadius:10,border:"1px solid var(--border)",fontSize:13,color:"var(--dim)",lineHeight:1.7}}>
            <strong style={{color:"var(--text)"}}>{cfg.salesRows.toLocaleString()}</strong> rows · <strong style={{color:"var(--text)"}}>{cfg.customers.toLocaleString()}</strong> cust · <strong style={{color:"var(--text)"}}>{cfg.products.toLocaleString()}</strong> prod · {cfg.startDate} {"\u2192"} {cfg.endDate} · <Badge>{cfg.format.toUpperCase()}</Badge>
            {cfg.salesOutput!=="sales"&&<>{" · "}<Badge>{cfg.salesOutput==="both"?"Sales + Orders":"Orders"}</Badge></>}
            {cfg.returnsEnabled&&<>{" · "}<Badge variant="success">Returns {(cfg.returnRate*100).toFixed(1)}%</Badge></>}
          </div>
          <div style={{marginTop:14}}>
            <button onClick={runGenerate} disabled={errors.length>0||isRunning} style={{width:"100%",padding:"12px 24px",borderRadius:10,border:"none",fontSize:14,fontWeight:700,cursor:errors.length>0?"not-allowed":"pointer",fontFamily:"var(--sans)",transition:"all .15s",background:errors.length>0?"var(--alt)":"var(--accent)",color:errors.length>0?"var(--muted)":"#fff",opacity:isRunning?.7:1,boxShadow:errors.length===0&&!isRunning?"0 4px 16px rgba(79,91,213,.2)":"none"}}>{isRunning?"Generating...":"Generate Data"}</button>
            {isRunning&&<button onClick={cancelGenerate} style={{width:"100%",padding:10,marginTop:8,borderRadius:10,border:"1px solid var(--err)",background:"var(--errBg)",color:"var(--err)",fontWeight:600,fontSize:13,cursor:"pointer",fontFamily:"var(--sans)"}}>Cancel</button>}
          </div>
          <LogViewer logs={logs} isRunning={isRunning} elapsed={elapsed} />
        </Section>
        </div>
        )}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
