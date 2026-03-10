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
  if(c.marginMax<=c.marginMin) w.push("Max margin should exceed min margin.");
  if(c.csEnabled&&c.csPerCustomerMax<c.csPerCustomerMin) e.push("Segment max per customer must be >= min.");
  if(c.csEnabled&&c.csPerCustomerMax>c.csSegmentCount) e.push("Segment max per customer must be <= segment count.");
  if(c.spEnabled&&c.spPerCustomerMax<c.spPerCustomerMin) e.push("Superpowers max per customer must be >= min.");
  if(c.spEnabled&&c.spPerCustomerMax>c.spPowersCount) e.push("Superpowers max per customer must be <= powers count.");
  if(c.employeeMaxStaff<c.employeeMinStaff) w.push("Employee max staff should be >= min staff.");
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

  /* Run history */
  const[runHistory,setRunHistory]=useState([]);

  /* Dimension tab */
  const[dimTab,setDimTab]=useState("customers");

  /* Sidebar collapse */
  const[sidebarOpen,setSidebarOpen]=useState(true);

  /* Preset preview */
  const[previewPreset,setPreviewPreset]=useState(null);
  const[previewData,setPreviewData]=useState(null);

  /* Models YAML state */
  const[modelsYaml,setModelsYaml]=useState("");
  const[modelsOrig,setModelsOrig]=useState("");
  const[modelsDisk,setModelsDisk]=useState("");
  const[modelsErr,setModelsErr]=useState(null);
  const modelsDirty=modelsYaml!==modelsOrig;
  const modelsApplied=modelsOrig!==modelsDisk;

  /* Models form state (visual editor) */
  const[modelsTab,setModelsTab]=useState("visual");
  const[mf,setMf]=useState(null);

  /* Config YAML state */
  const[cfgYaml,setCfgYaml]=useState("");
  const[cfgYamlOrig,setCfgYamlOrig]=useState("");
  const[cfgYamlErr,setCfgYamlErr]=useState(null);
  const cfgYamlDirty=cfgYaml!==cfgYamlOrig;

  /* Page navigation */
  const[page,setPage_]=useState("main");
  const setPage=useCallback((p)=>{
    document.activeElement?.blur();
    const scrollY=window.scrollY;
    setPage_(p);
    if(p==="config")loadCfgYaml();
    if(p==="models"){fetch(API+"/models").then(r=>r.text()).then(t=>{setModelsYaml(t);setModelsOrig(t);}).catch(()=>{});fetch(API+"/models/form").then(r=>r.json()).then(d=>setMf(d)).catch(()=>{});}
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
      setCfg({seed:42,format:"parquet",salesOutput:"sales",skipOrderCols:false,compression:"snappy",rowGroupSize:2000000,mergeParquet:true,partitionEnabled:true,maxLinesPerOrder:5,salesOptimize:true,startDate:"2020-01-01",endDate:"2025-12-31",fiscalMonthOffset:0,asOfDate:"",includeCalendar:true,includeIso:false,includeFiscal:true,includeWeeklyFiscal:false,wfFirstDay:0,wfWeeklyType:"Last",wfQuarterType:"445",wfTypeStartFiscalYear:1,salesRows:103285,chunkSize:1000000,autoWorkers:false,workers:8,customers:48837,stores:10,products:2581,promotions:20,pctIndia:10,pctUs:51,pctEu:39,pctAsia:0,pctOrg:1,customerActiveRatio:.98,profile:"steady",firstYearPct:.27,valueScale:1,minPrice:10,maxPrice:3000,productActiveRatio:.98,marginMin:.20,marginMax:.35,brandNormalize:false,brandNormalizeAlpha:.35,geoWeights:{"United States":.35,India:.2,"United Kingdom":.1,Germany:.1,France:.1,Australia:.07,Canada:.08},returnsEnabled:true,returnRate:.03,returnMinDays:1,returnMaxDays:60,promoNewCustWindow:3,csEnabled:false,csGenerateBridge:false,csSegmentCount:10,csPerCustomerMin:1,csPerCustomerMax:2,csIncludeScore:true,csIncludePrimaryFlag:true,csIncludeValidity:true,csValidityGrain:"month",csChurnRateQtr:.08,csNewCustomerMonths:2,csSeed:123,spEnabled:false,spGenerateBridge:false,spPowersCount:20,spPerCustomerMin:1,spPerCustomerMax:3,spIncludePowerLevel:true,spIncludePrimaryFlag:true,spIncludeAcquiredDate:true,spIncludeValidity:false,spSeed:123,storeEnsureIsoCoverage:true,storeDistrictSize:10,storeDistrictsPerRegion:8,storeOpeningStart:"1995-01-01",storeOpeningEnd:"2023-12-31",storeClosingEnd:"2028-12-31",storeAssortmentEnabled:true,employeeMinStaff:3,employeeMaxStaff:5,employeeEmailDomain:"contoso.com",employeeStoreAssignments:true,erCurrencies:["CAD","GBP","EUR","INR","AUD","CNY","JPY"],erBaseCurrency:"USD",erVolatility:.02,erFutureDrift:.02,erUseGlobalDates:true,budgetEnabled:true,budgetReportCurrency:"USD",budgetDefaultGrowth:.05,budgetReturnRateCap:.30,inventoryEnabled:true,inventoryGrain:"monthly",inventoryShrinkageEnabled:true,inventoryShrinkageRate:.02,regenAll:false,regenDims:{}});
    });
    fetch(API+"/presets").then(r=>r.json()).then(d=>{setPresets(d);setPresetBucket(Object.keys(d)[0]||"");}).catch(()=>{});
    fetch(API+"/models").then(r=>r.text()).then(t=>{setModelsYaml(t);setModelsOrig(t);setModelsDisk(t);}).catch(()=>{});
    fetch(API+"/models/form").then(r=>r.json()).then(d=>setMf(d)).catch(()=>{});
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
    fetch(API+"/config/yaml/disk").then(r=>r.text()).then(t=>{
      setCfgYaml(t);setCfgYamlOrig(t);setCfgYamlErr(null);
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
          if(d.type==="end"){
            clearInterval(timerRef.current);setIsRunning(false);setElapsed(d.elapsed);es.close();
            const fmtE=d.elapsed<60?`${d.elapsed.toFixed(1)}s`:`${Math.floor(d.elapsed/60)}m ${Math.floor(d.elapsed%60)}s`;
            flash(d.status==="done"?`Pipeline completed in ${fmtE}`:`Pipeline failed after ${fmtE}`);
            setRunHistory(p=>[{status:d.status,elapsed:d.elapsed,rows:cfg.salesRows,time:new Date().toLocaleTimeString()},...p].slice(0,10));
          }
          if(d.type==="idle"){es.close();}
        };
        es.onerror=()=>{clearInterval(timerRef.current);setIsRunning(false);es.close();};
      }).catch(e=>{setLogs(["Failed to start pipeline: "+e.message]);});
  };

  const cancelGenerate=()=>{
    fetch(API+"/generate/cancel",{method:"POST"}).then(()=>{clearInterval(timerRef.current);setIsRunning(false);}).catch(()=>{});
  };

  /* ─── Ctrl+Enter to generate ─── */
  useEffect(()=>{
    const handler=(e)=>{
      if((e.ctrlKey||e.metaKey)&&e.key==="Enter"){e.preventDefault();runGenerate();}
    };
    window.addEventListener("keydown",handler);
    return()=>window.removeEventListener("keydown",handler);
  });

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
        fetch(API+"/models/form").then(r=>r.json()).then(d=>setMf(d)).catch(()=>{});
      }).catch(()=>{});
  };

  /* ─── Models form sync ─── */
  const syncModelsForm=useDebounce((vals)=>{
    fetch(API+"/models/form",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({values:vals})})
      .then(r=>r.json()).then(d=>{if(d.yaml_text){setModelsYaml(d.yaml_text);setModelsOrig(d.yaml_text);}}).catch(()=>{});
  },400);
  const sm=useCallback((k,v)=>{
    setMf(p=>{const n={...p,[k]:v};syncModelsForm(n);return n;});
  },[syncModelsForm]);

  /* ─── Config YAML ─── */
  const saveCfgYaml=()=>{
    setCfgYamlErr(null);
    fetch(API+"/config/yaml",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({yaml_text:cfgYaml})})
      .then(r=>{if(!r.ok)return r.json().then(d=>{throw new Error(d.detail||"Save failed")});return r.json();})
      .then(()=>{setCfgYamlOrig(cfgYaml);setCfgYamlErr(null);flash("Config applied (in memory)");reloadFormFromServer();})
      .catch(e=>setCfgYamlErr(e.message));
  };

  const resetCfgYaml=()=>{
    fetch(API+"/config/yaml/reset",{method:"POST"})
      .then(r=>r.json())
      .then(()=>{loadCfgYaml();reloadFormFromServer();flash("Config reset to disk version");})
      .catch(()=>{});
  };

  const refreshCfgYamlFromUI=()=>{
    fetch(API+"/config/yaml").then(r=>r.text()).then(t=>{setCfgYaml(t);setCfgYamlOrig(t);setCfgYamlErr(null);flash("Loaded current UI settings");}).catch(()=>{});
  };

  /* ─── Render ─── */
  if(!cfg)return <div style={{padding:40,textAlign:"center",color:"var(--muted)"}}>Loading configuration...</div>;
  const{errors,warnings}=validate(cfg);
  const showPq=cfg.format==="parquet"||cfg.format==="deltaparquet";
  const showDelta=cfg.format==="deltaparquet";

  /* ─── Config form sections (used in main page) ─── */
  const renderConfigForm=()=>(<>
        {/* 1 OUTPUT */}
        <Section num="1" title="Output">
          <R2>
            <F label="Output format"><Sel value={cfg.format} onChange={v=>s("format",v)} options={["parquet","csv","deltaparquet"]} labels={["Parquet","CSV","Delta (Parquet)"]} /></F>
            <F label="Sales output"><Sel value={cfg.salesOutput} onChange={v=>s("salesOutput",v)} options={["sales","sales_order","both"]} labels={["Sales (flat)","Order Header + Detail","Both"]} /></F>
          </R2>
          <R3>
            <F label="Max lines per order"><N value={cfg.maxLinesPerOrder} onChange={v=>s("maxLinesPerOrder",v)} min={1} max={20} step={1} /></F>
            <F label="Seed" help="Global random seed for reproducibility."><N value={cfg.seed} onChange={v=>s("seed",v)} min={0} step={1} /></F>
            <F label=" "><div style={{display:"flex",flexDirection:"column",gap:8,paddingTop:16}}><Check checked={cfg.skipOrderCols} onChange={v=>s("skipOrderCols",v)} label="Skip order columns" /><Check checked={cfg.salesOptimize} onChange={v=>s("salesOptimize",v)} label="Optimize merged files" /></div></F>
          </R3>
        </Section>

        {/* 2 DATES */}
        <Section num="2" title="Dates">
          <R3>
            <F label="Start date"><input type="date" style={iS} value={cfg.startDate} onChange={e=>s("startDate",e.target.value)} /></F>
            <F label="End date"><input type="date" style={iS} value={cfg.endDate} onChange={e=>s("endDate",e.target.value)} /></F>
            <F label="As-of date" help="Optional override for 'today' in lifecycle logic. Leave blank for system date."><input type="date" style={iS} value={cfg.asOfDate||""} onChange={e=>s("asOfDate",e.target.value)} /></F>
          </R3>
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
            {cfg.includeWeeklyFiscal&&<div style={{marginTop:10}}><F label="Type start fiscal year" help="Controls which fiscal year a week belongs to when it spans two years."><Sel value={String(cfg.wfTypeStartFiscalYear)} onChange={v=>s("wfTypeStartFiscalYear",parseInt(v))} options={["1","2"]} labels={["Type 1 (week belongs to year it starts in)","Type 2 (week belongs to year it ends in)"]} /></F></div>}
          </Box>
        </Section>

        {/* 3 SCALE */}
        <Section num="3" title="Scale">
          <F label="Sales rows" help="Total rows to generate in the Sales fact table."><N value={cfg.salesRows} onChange={v=>s("salesRows",v)} min={1} step={10000} /></F>
          <R4>
            <F label="Customers"><N value={cfg.customers} onChange={v=>s("customers",v)} min={1} step={1000} /></F>
            <F label="Products"><N value={cfg.products} onChange={v=>s("products",v)} min={1} step={500} /></F>
            <F label="Stores"><N value={cfg.stores} onChange={v=>s("stores",v)} min={1} step={10} /></F>
            <F label="Promotions"><N value={cfg.promotions} onChange={v=>s("promotions",v)} min={0} step={5} /></F>
          </R4>
        </Section>

        {/* 5 DIMENSIONS & FEATURES — tabbed panel */}
        <Section num="4" title="Dimensions & Features">
          <div style={{display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:6,marginTop:6,marginBottom:4}}>
            {[
              ["customers","Customers"],["products","Products"],["stores","Stores"],["employees","Employees"],
              ["returns","Returns",cfg.returnsEnabled],["promotions","Promos"],
              ["segments","Segments",cfg.csEnabled],["superpowers","Powers",cfg.spEnabled],
              ["exchange","FX Rates"],["budget","Budget",cfg.budgetEnabled],["inventory","Inventory",cfg.inventoryEnabled],
            ].map(([k,label,flag])=>{
              const active=dimTab===k;
              return(
                <button key={k} onClick={()=>setDimTab(k)}
                  style={{padding:"5px 0",borderRadius:6,fontSize:11.5,fontWeight:active?600:500,cursor:"pointer",fontFamily:"var(--sans)",transition:"all .12s",whiteSpace:"nowrap",display:"flex",alignItems:"center",justifyContent:"center",gap:5,border:active?"1px solid var(--accent)":"1px solid var(--border)",background:active?"var(--glow)":"transparent",color:active?"var(--accent)":"var(--dim)"}}
                  onMouseOver={e=>{if(!active){e.currentTarget.style.background="var(--alt)";e.currentTarget.style.color="var(--text)";}}}
                  onMouseOut={e=>{if(!active){e.currentTarget.style.background="transparent";e.currentTarget.style.color="var(--dim)";}}}
                >
                  {label}
                  {flag!==undefined&&<span style={{width:6,height:6,borderRadius:"50%",background:flag?"var(--ok)":"var(--muted)",flexShrink:0}} />}
                </button>
              );
            })}
          </div>

          {/* ── Customers ── */}
          {dimTab==="customers"&&<div style={{marginTop:8}}>
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
          </div>}

          {/* ── Products ── */}
          {dimTab==="products"&&<div style={{marginTop:8}}>
            <Box title="Pricing">
              <R3>
                <F label="Value scale" help="Multiplier on base product prices."><N value={cfg.valueScale} onChange={v=>s("valueScale",v)} min={.01} max={10} step={.05} /></F>
                <F label="Min unit price"><N value={cfg.minPrice} onChange={v=>s("minPrice",v)} min={0} step={10} /></F>
                <F label="Max unit price"><N value={cfg.maxPrice} onChange={v=>s("maxPrice",v)} min={1} step={50} /></F>
              </R3>
              {cfg.maxPrice>cfg.minPrice&&<div style={{fontSize:11,color:"var(--muted)",marginTop:8}}>Scaled range: ~{((cfg.minPrice||0)*(cfg.valueScale||1)).toLocaleString()} {"\u2192"} {((cfg.maxPrice||0)*(cfg.valueScale||1)).toLocaleString()}</div>}
            </Box>
            <Box title="Cost Margins">
              <R2>
                <F label="Min margin %" help="Minimum cost margin as a fraction (e.g. 0.20 = 20%)."><N value={cfg.marginMin} onChange={v=>s("marginMin",v)} min={0} max={1} step={.01} /></F>
                <F label="Max margin %" help="Maximum cost margin as a fraction."><N value={cfg.marginMax} onChange={v=>s("marginMax",v)} min={0} max={1} step={.01} /></F>
              </R2>
              <div style={{fontSize:11,color:"var(--muted)",marginTop:6}}>Margin range: {((cfg.marginMin||0)*100).toFixed(0)}% {"\u2013"} {((cfg.marginMax||0)*100).toFixed(0)}%</div>
            </Box>
            <Box title="Brand Normalization">
              <Check checked={cfg.brandNormalize} onChange={v=>s("brandNormalize",v)} label="Pull brand prices toward global median" />
              {cfg.brandNormalize&&<div style={{marginTop:8}}><Sld label="Alpha (brand identity retention)" value={cfg.brandNormalizeAlpha} min={0} max={1} step={.05} onChange={v=>s("brandNormalizeAlpha",v)} /></div>}
            </Box>
            <Sld label="Active ratio" value={cfg.productActiveRatio} min={.1} max={1} step={.01} onChange={v=>s("productActiveRatio",v)} />
          </div>}

          {/* ── Stores ── */}
          {dimTab==="stores"&&<div style={{marginTop:8}}>
            <R3>
              <F label="District size" help="Stores per district."><N value={cfg.storeDistrictSize} onChange={v=>s("storeDistrictSize",v)} min={1} step={1} /></F>
              <F label="Districts per region"><N value={cfg.storeDistrictsPerRegion} onChange={v=>s("storeDistrictsPerRegion",v)} min={1} step={1} /></F>
              <F label=" "><div style={{paddingTop:16}}><Check checked={cfg.storeEnsureIsoCoverage} onChange={v=>s("storeEnsureIsoCoverage",v)} label="Ensure ISO country coverage" /></div></F>
            </R3>
            <Box title="Store opening/closing dates">
              <R3>
                <F label="Opening start"><input type="date" style={iS} value={cfg.storeOpeningStart} onChange={e=>s("storeOpeningStart",e.target.value)} /></F>
                <F label="Opening end"><input type="date" style={iS} value={cfg.storeOpeningEnd} onChange={e=>s("storeOpeningEnd",e.target.value)} /></F>
                <F label="Closing end"><input type="date" style={iS} value={cfg.storeClosingEnd} onChange={e=>s("storeClosingEnd",e.target.value)} /></F>
              </R3>
            </Box>
            <Box title="Assortment">
              <Check checked={cfg.storeAssortmentEnabled} onChange={v=>s("storeAssortmentEnabled",v)} label="Enable product assortment filtering" />
              <div style={{fontSize:11,color:"var(--muted)",marginTop:4}}>When off, every store sells every product (cross-join).</div>
            </Box>
          </div>}

          {/* ── Employees ── */}
          {dimTab==="employees"&&<div style={{marginTop:8}}>
            <R3>
              <F label="Min staff per store"><N value={cfg.employeeMinStaff} onChange={v=>s("employeeMinStaff",v)} min={1} step={1} /></F>
              <F label="Max staff per store"><N value={cfg.employeeMaxStaff} onChange={v=>s("employeeMaxStaff",v)} min={1} step={1} /></F>
              <F label="Email domain"><input type="text" style={iS} value={cfg.employeeEmailDomain} onChange={e=>s("employeeEmailDomain",e.target.value)} /></F>
            </R3>
            <div style={{marginTop:12}}><Check checked={cfg.employeeStoreAssignments} onChange={v=>s("employeeStoreAssignments",v)} label="Enable store assignments (role-based scheduling)" /></div>
          </div>}

          {/* ── Returns ── */}
          {dimTab==="returns"&&<div style={{marginTop:8}}>
            <div style={{marginTop:4}}><Check checked={cfg.returnsEnabled} onChange={v=>s("returnsEnabled",v)} label="Enable returns generation" /></div>
            {cfg.returnsEnabled&&<R3>
              <F label="Return rate" help="Fraction of sales rows returned."><N value={cfg.returnRate} onChange={v=>s("returnRate",v)} min={0} max={1} step={.005} /></F>
              <F label="Min days after sale"><N value={cfg.returnMinDays} onChange={v=>s("returnMinDays",v)} min={1} step={1} /></F>
              <F label="Max days after sale"><N value={cfg.returnMaxDays} onChange={v=>s("returnMaxDays",v)} min={1} step={5} /></F>
            </R3>}
          </div>}

          {/* ── Promotions ── */}
          {dimTab==="promotions"&&<div style={{marginTop:8}}>
            <Box title="Promotion counts by type">
              <R4>
                <F label="Seasonal" help="Seasonal discount promos per year window."><N value={cfg.promoSeasonal} onChange={v=>s("promoSeasonal",v)} min={0} max={100} step={1} /></F>
                <F label="Clearance" help="Clearance sale promos (steep discounts)."><N value={cfg.promoClearance} onChange={v=>s("promoClearance",v)} min={0} max={100} step={1} /></F>
                <F label="Limited Time" help="Limited-time offer promos."><N value={cfg.promoLimited} onChange={v=>s("promoLimited",v)} min={0} max={100} step={1} /></F>
                <F label="Flash Sale" help="Short 1–2 day flash sales."><N value={cfg.promoFlash} onChange={v=>s("promoFlash",v)} min={0} max={100} step={1} /></F>
              </R4>
              <R4>
                <F label="Volume" help="Volume/bulk discount promos."><N value={cfg.promoVolume} onChange={v=>s("promoVolume",v)} min={0} max={100} step={1} /></F>
                <F label="Loyalty" help="Loyalty-exclusive promos."><N value={cfg.promoLoyalty} onChange={v=>s("promoLoyalty",v)} min={0} max={100} step={1} /></F>
                <F label="Bundle" help="Bundle deal promos."><N value={cfg.promoBundle} onChange={v=>s("promoBundle",v)} min={0} max={100} step={1} /></F>
                <F label="New Customer" help="New customer welcome promos."><N value={cfg.promoNewCustomer} onChange={v=>s("promoNewCustomer",v)} min={0} max={100} step={1} /></F>
              </R4>
              <div style={{fontSize:11,color:"var(--muted)",marginTop:6}}>Total: {(cfg.promoSeasonal||0)+(cfg.promoClearance||0)+(cfg.promoLimited||0)+(cfg.promoFlash||0)+(cfg.promoVolume||0)+(cfg.promoLoyalty||0)+(cfg.promoBundle||0)+(cfg.promoNewCustomer||0)} promos (+ holidays auto-generated per year)</div>
            </Box>
            <F label="New customer window (months)" help="Months after CustomerStartDate where New Customer promo applies. 0 = same month only."><N value={cfg.promoNewCustWindow} onChange={v=>s("promoNewCustWindow",v)} min={0} max={24} step={1} /></F>
          </div>}

          {/* ── Customer Segments ── */}
          {dimTab==="segments"&&<div style={{marginTop:8}}>
            <div style={{marginTop:4}}><Check checked={cfg.csEnabled} onChange={v=>s("csEnabled",v)} label="Enable customer segments" /></div>
            {cfg.csEnabled&&<>
              <div style={{marginTop:8}}><Check checked={cfg.csGenerateBridge} onChange={v=>s("csGenerateBridge",v)} label="Generate bridge table" /></div>
              <R3>
                <F label="Segment count"><N value={cfg.csSegmentCount} onChange={v=>s("csSegmentCount",v)} min={1} step={1} /></F>
                <F label="Min per customer"><N value={cfg.csPerCustomerMin} onChange={v=>s("csPerCustomerMin",v)} min={1} step={1} /></F>
                <F label="Max per customer"><N value={cfg.csPerCustomerMax} onChange={v=>s("csPerCustomerMax",v)} min={1} step={1} /></F>
              </R3>
              <Box title="Include columns">
                <div style={{display:"flex",gap:16,flexWrap:"wrap"}}>
                  <Check checked={cfg.csIncludeScore} onChange={v=>s("csIncludeScore",v)} label="Score" />
                  <Check checked={cfg.csIncludePrimaryFlag} onChange={v=>s("csIncludePrimaryFlag",v)} label="Primary flag" />
                  <Check checked={cfg.csIncludeValidity} onChange={v=>s("csIncludeValidity",v)} label="Validity periods" />
                </div>
              </Box>
              {cfg.csIncludeValidity&&<Box title="Validity settings">
                <R3>
                  <F label="Grain"><Sel value={cfg.csValidityGrain} onChange={v=>s("csValidityGrain",v)} options={["month","day"]} /></F>
                  <F label="Churn rate (quarterly)"><N value={cfg.csChurnRateQtr} onChange={v=>s("csChurnRateQtr",v)} min={0} max={1} step={.01} /></F>
                  <F label="New customer months"><N value={cfg.csNewCustomerMonths} onChange={v=>s("csNewCustomerMonths",v)} min={0} step={1} /></F>
                </R3>
              </Box>}
              <F label="Seed"><N value={cfg.csSeed} onChange={v=>s("csSeed",v)} min={0} step={1} /></F>
            </>}
          </div>}

          {/* ── Superpowers ── */}
          {dimTab==="superpowers"&&<div style={{marginTop:8}}>
            <div style={{marginTop:4}}><Check checked={cfg.spEnabled} onChange={v=>s("spEnabled",v)} label="Enable superpowers" /></div>
            {cfg.spEnabled&&<>
              <div style={{marginTop:8}}><Check checked={cfg.spGenerateBridge} onChange={v=>s("spGenerateBridge",v)} label="Generate bridge table" /></div>
              <R3>
                <F label="Powers count"><N value={cfg.spPowersCount} onChange={v=>s("spPowersCount",v)} min={1} step={1} /></F>
                <F label="Min per customer"><N value={cfg.spPerCustomerMin} onChange={v=>s("spPerCustomerMin",v)} min={1} step={1} /></F>
                <F label="Max per customer"><N value={cfg.spPerCustomerMax} onChange={v=>s("spPerCustomerMax",v)} min={1} step={1} /></F>
              </R3>
              <Box title="Include columns">
                <div style={{display:"flex",gap:16,flexWrap:"wrap"}}>
                  <Check checked={cfg.spIncludePowerLevel} onChange={v=>s("spIncludePowerLevel",v)} label="Power level" />
                  <Check checked={cfg.spIncludePrimaryFlag} onChange={v=>s("spIncludePrimaryFlag",v)} label="Primary flag" />
                  <Check checked={cfg.spIncludeAcquiredDate} onChange={v=>s("spIncludeAcquiredDate",v)} label="Acquired date" />
                  <Check checked={cfg.spIncludeValidity} onChange={v=>s("spIncludeValidity",v)} label="Validity periods" />
                </div>
              </Box>
              <F label="Seed"><N value={cfg.spSeed} onChange={v=>s("spSeed",v)} min={0} step={1} /></F>
            </>}
          </div>}

          {/* ── Exchange Rates ── */}
          {dimTab==="exchange"&&<div style={{marginTop:8}}>
            <F label="Currencies" help="Comma-separated list of currency codes.">
              <input type="text" style={iS} value={(cfg.erCurrencies||[]).join(", ")} onChange={e=>s("erCurrencies",e.target.value.split(",").map(c=>c.trim()).filter(Boolean))} />
            </F>
            <R3>
              <F label="Base currency"><input type="text" style={iS} value={cfg.erBaseCurrency} onChange={e=>s("erBaseCurrency",e.target.value.toUpperCase())} /></F>
              <F label="Volatility" help="Daily FX rate volatility."><N value={cfg.erVolatility} onChange={v=>s("erVolatility",v)} min={0} max={.5} step={.005} /></F>
              <F label="Future annual drift"><N value={cfg.erFutureDrift} onChange={v=>s("erFutureDrift",v)} min={0} max={.5} step={.005} /></F>
            </R3>
            <div style={{marginTop:10}}><Check checked={cfg.erUseGlobalDates} onChange={v=>s("erUseGlobalDates",v)} label="Use global date range" /></div>
          </div>}

          {/* ── Budget ── */}
          {dimTab==="budget"&&<div style={{marginTop:8}}>
            <div style={{marginTop:4}}><Check checked={cfg.budgetEnabled} onChange={v=>s("budgetEnabled",v)} label="Generate Budget fact table" /></div>
            {cfg.budgetEnabled&&<R3>
              <F label="Report currency"><input type="text" style={iS} value={cfg.budgetReportCurrency} onChange={e=>s("budgetReportCurrency",e.target.value.toUpperCase())} /></F>
              <F label="Default backcast growth"><N value={cfg.budgetDefaultGrowth} onChange={v=>s("budgetDefaultGrowth",v)} min={-1} max={1} step={.01} /></F>
              <F label="Return rate cap"><N value={cfg.budgetReturnRateCap} onChange={v=>s("budgetReturnRateCap",v)} min={0} max={1} step={.01} /></F>
            </R3>}
          </div>}

          {/* ── Inventory ── */}
          {dimTab==="inventory"&&<div style={{marginTop:8}}>
            <div style={{marginTop:4}}><Check checked={cfg.inventoryEnabled} onChange={v=>s("inventoryEnabled",v)} label="Generate Inventory Snapshot fact table" /></div>
            {cfg.inventoryEnabled&&<>
              <R2>
                <F label="Grain"><Sel value={cfg.inventoryGrain} onChange={v=>s("inventoryGrain",v)} options={["monthly","daily"]} /></F>
                <F label=" "><div style={{paddingTop:16}}><Check checked={cfg.inventoryShrinkageEnabled} onChange={v=>s("inventoryShrinkageEnabled",v)} label="Enable shrinkage" /></div></F>
              </R2>
              {cfg.inventoryShrinkageEnabled&&<Sld label="Shrinkage rate" value={cfg.inventoryShrinkageRate} min={0} max={.1} step={.005} onChange={v=>s("inventoryShrinkageRate",v)} fmt={v=>`${(v*100).toFixed(1)}%`} />}
            </>}
          </div>}
        </Section>

        {/* 5 ADVANCED */}
        <Section num="5" title="Advanced" defaultOpen={false}>
          <Box title="Pipeline tuning">
            <R3>
              <F label="Chunk size" help="Rows per parallel chunk."><N value={cfg.chunkSize} onChange={v=>s("chunkSize",v)} min={10000} step={100000} /></F>
              <F label="Workers"><div style={{display:"flex",alignItems:"center",gap:10}}><Check checked={cfg.autoWorkers} onChange={v=>{s("autoWorkers",v);if(v)s("workers",0);}} label="Auto" />{!cfg.autoWorkers&&<N value={cfg.workers} onChange={v=>s("workers",v)} min={1} max={32} style={{width:100}} />}</div></F>
              <F label=" " />
            </R3>
          </Box>
          {showPq&&<Box title="Parquet options">
            <R3>
              <F label="Compression"><Sel value={cfg.compression} onChange={v=>s("compression",v)} options={["snappy","zstd","gzip","none"]} /></F>
              <F label="Row group size"><N value={cfg.rowGroupSize} onChange={v=>s("rowGroupSize",v)} min={100000} step={500000} /></F>
              <F label=" "><div style={{display:"flex",flexDirection:"column",gap:8,paddingTop:3}}><Check checked={cfg.mergeParquet} onChange={v=>s("mergeParquet",v)} label="Merge parquet chunks" />{showDelta&&<Check checked={cfg.partitionEnabled} onChange={v=>s("partitionEnabled",v)} label="Partition by Year/Month" />}</div></F>
            </R3>
          </Box>}
        </Section>

  </>);

  const sidebarW=sidebarOpen?360:0;

  return(
    <div className="app-layout" style={{display:"flex",minHeight:"100vh"}}>
      {toast&&<div style={{position:"fixed",top:16,right:16,zIndex:999,background:toast.includes("failed")?"var(--err)":"var(--accent)",color:"#fff",padding:"9px 20px",borderRadius:8,fontSize:13,fontWeight:600,boxShadow:"0 6px 24px var(--shadow)",animation:"slideIn .28s ease"}}>{toast}</div>}

      {/* ═══ SIDEBAR ═══ */}
      <div className="app-sidebar" style={{width:sidebarW,flexShrink:0,background:"var(--surface)",borderRight:sidebarOpen?"1px solid var(--border)":"none",overflowY:"auto",overflowX:"hidden",transition:"width .2s",position:"relative"}}>
        {sidebarOpen&&<div style={{padding:20,width:360}}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:10}}>
          <div style={{fontSize:10.5,fontWeight:700,color:"var(--muted)",letterSpacing:".09em",textTransform:"uppercase"}}>Presets</div>
          <div style={{display:"flex",gap:6}}><ThemeToggle /><button onClick={()=>setSidebarOpen(false)} title="Collapse sidebar" style={{width:32,height:32,borderRadius:8,border:"1px solid var(--border)",background:"var(--surface)",color:"var(--dim)",fontSize:14,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center"}} onMouseOver={e=>{e.currentTarget.style.borderColor="var(--accent)";}} onMouseOut={e=>{e.currentTarget.style.borderColor="var(--border)";}}>{"\u2039"}</button></div>
        </div>
        {Object.keys(presets).length>0?(<>
          <Sel value={presetBucket} onChange={setPresetBucket} options={Object.keys(presets)} />
          <div style={{marginTop:8,display:"flex",flexDirection:"column",gap:4}}>{Object.entries(presets[presetBucket]||{}).map(([name,p])=>{
            const sel=selectedPreset===name;
            const c=p||{};const cust=c.customers||c.total_customers;const prod=c.products||c.num_products;const rows=c.sales_rows||c.total_rows;
            return(
              <button key={name} onClick={()=>applyPreset(name)}
                onMouseEnter={()=>setPreviewPreset(name)}
                onMouseLeave={()=>setPreviewPreset(null)}
                style={{display:"flex",flexDirection:"column",alignItems:"flex-start",width:"100%",padding:"9px 12px",borderRadius:8,cursor:"pointer",border:`1px solid ${sel?"var(--accent)":"var(--border)"}`,background:sel?"var(--glow)":"var(--surface)",fontFamily:"var(--sans)",transition:"all .12s",gap:4,position:"relative"}}>
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

        {/* Preset preview tooltip */}
        {previewPreset&&presets[presetBucket]&&presets[presetBucket][previewPreset]&&(()=>{
          const p=presets[presetBucket][previewPreset];
          const changes=[];
          if(p.sales_rows||p.total_rows){const v=p.sales_rows||p.total_rows;if(v!==cfg.salesRows)changes.push({k:"Sales rows",from:cfg.salesRows.toLocaleString(),to:v.toLocaleString()});}
          if(p.customers||p.total_customers){const v=p.customers||p.total_customers;if(v!==cfg.customers)changes.push({k:"Customers",from:cfg.customers.toLocaleString(),to:v.toLocaleString()});}
          if(p.products||p.num_products){const v=p.products||p.num_products;if(v!==cfg.products)changes.push({k:"Products",from:cfg.products.toLocaleString(),to:v.toLocaleString()});}
          if(p.stores&&p.stores!==cfg.stores)changes.push({k:"Stores",from:String(cfg.stores),to:String(p.stores)});
          if(changes.length===0)return null;
          return(
            <div style={{marginTop:8,padding:"8px 10px",borderRadius:6,background:"var(--alt)",border:"1px solid var(--border)",fontSize:11,animation:"fadeIn .15s ease"}}>
              <div style={{fontWeight:600,color:"var(--dim)",marginBottom:4}}>Preview changes:</div>
              {changes.map(c=>(
                <div key={c.k} style={{display:"flex",justifyContent:"space-between",padding:"2px 0",color:"var(--text)"}}>
                  <span>{c.k}</span>
                  <span><span style={{color:"var(--err)",textDecoration:"line-through"}}>{c.from}</span>{" \u2192 "}<span style={{color:"var(--ok)",fontWeight:600}}>{c.to}</span></span>
                </div>
              ))}
            </div>
          );
        })()}

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
          {cfgYamlDirty&&<span style={{width:7,height:7,borderRadius:"50%",background:"var(--warn)",flexShrink:0}} />}
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

        {/* Validation */}
        {cfg&&<>
          <div style={{borderTop:"1px solid var(--border)",margin:"20px 0"}} />
          <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:8}}>
            <div style={{fontSize:10.5,fontWeight:700,color:"var(--muted)",letterSpacing:".09em",textTransform:"uppercase"}}>Validation</div>
            <Badge variant={errors.length>0?"error":warnings.length>0?"warning":"success"}>{errors.length>0?`${errors.length} error${errors.length>1?"s":""}`:warnings.length>0?`${warnings.length} warning${warnings.length>1?"s":""}`:"Valid"}</Badge>
          </div>
          {errors.length===0&&warnings.length===0&&<div style={{fontSize:11.5,color:"var(--dim)",display:"flex",alignItems:"center",gap:6}}><span style={{color:"var(--ok)"}}>&#10003;</span> Configuration looks good.</div>}
          {errors.map((e,i)=><div key={`e${i}`} style={{padding:"7px 10px",borderRadius:6,marginTop:4,fontSize:11.5,display:"flex",alignItems:"flex-start",gap:6,background:"var(--errBg)",color:"var(--err)"}}><span style={{fontWeight:700,flexShrink:0}}>&#10005;</span><span>{e}</span></div>)}
          {warnings.map((w,i)=><div key={`w${i}`} style={{padding:"7px 10px",borderRadius:6,marginTop:4,fontSize:11.5,display:"flex",alignItems:"flex-start",gap:6,background:"var(--warnBg)",color:"var(--warn)"}}><span style={{fontWeight:700,flexShrink:0}}>&#9888;</span><span>{w}</span></div>)}
        </>}

        {/* Regenerate Dimensions */}
        {cfg&&<>
          <div style={{borderTop:"1px solid var(--border)",margin:"20px 0"}} />
          <div style={{fontSize:10.5,fontWeight:700,color:"var(--muted)",letterSpacing:".09em",textTransform:"uppercase",marginBottom:8}}>Regenerate</div>
          <div style={{marginBottom:8}}><Check checked={cfg.regenAll} onChange={v=>s("regenAll",v)} label="All dimensions" /></div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>{DIMS.map(d=><Check key={d} checked={cfg.regenAll||!!cfg.regenDims[d]} onChange={v=>s("regenDims",{...cfg.regenDims,[d]:v})} label={d.replace("_"," ")} disabled={cfg.regenAll} />)}</div>
        </>}

        {/* Run history */}
        {runHistory.length>0&&<>
          <div style={{borderTop:"1px solid var(--border)",margin:"20px 0"}} />
          <div style={{fontSize:10.5,fontWeight:700,color:"var(--muted)",letterSpacing:".09em",textTransform:"uppercase",marginBottom:8}}>Run History</div>
          {runHistory.map((r,i)=>(
            <div key={i} className="run-history-item" style={{display:"flex",alignItems:"center",gap:8,padding:"6px 10px",borderRadius:6,fontSize:11.5,marginBottom:2}}>
              <span style={{width:7,height:7,borderRadius:"50%",background:r.status==="done"?"var(--ok)":"var(--err)",flexShrink:0}} />
              <span style={{color:"var(--text)",fontFamily:"var(--mono)",flex:1}}>{r.rows>=1000000?(r.rows/1000000).toFixed(1)+"M":(r.rows/1000).toFixed(r.rows%1000?1:0)+"K"} rows</span>
              <span style={{color:"var(--accent)",fontFamily:"var(--mono)"}}>{r.elapsed<60?r.elapsed.toFixed(1)+"s":Math.floor(r.elapsed/60)+"m "+Math.floor(r.elapsed%60)+"s"}</span>
              <span style={{color:"var(--muted)",fontSize:10}}>{r.time}</span>
            </div>
          ))}
        </>}
        </div>}
      </div>

      {/* Sidebar collapsed toggle */}
      {!sidebarOpen&&<div style={{position:"fixed",top:12,left:12,zIndex:100,display:"flex",gap:6}}>
        <button onClick={()=>setSidebarOpen(true)} title="Open sidebar" style={{width:36,height:36,borderRadius:8,border:"1px solid var(--border)",background:"var(--surface)",color:"var(--dim)",fontSize:16,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center",boxShadow:"0 2px 8px var(--shadow)"}} onMouseOver={e=>{e.currentTarget.style.borderColor="var(--accent)";}} onMouseOut={e=>{e.currentTarget.style.borderColor="var(--border)";}}>{"\u203A"}</button>
        <ThemeToggle />
      </div>}

      {/* ═══ MAIN ═══ */}
      <div className="main-content" style={{flex:1,padding:"28px 40px",maxWidth:1120,overflowY:"auto",paddingBottom:page==="main"?80:28}}>

        {page==="models"?(
        /* ═══ MODELS EDITOR PAGE ═══ */
        <div>
          <button onClick={()=>setPage("main")} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:7,border:"1px solid var(--border)",background:"var(--surface)",color:"var(--dim)",fontSize:12.5,cursor:"pointer",fontFamily:"var(--sans)",marginBottom:18,transition:"all .12s"}} onMouseOver={e=>e.currentTarget.style.borderColor="var(--accent)"} onMouseOut={e=>e.currentTarget.style.borderColor="var(--border)"}>
            {"\u2190"} Back to configuration
          </button>
          <h1 style={{fontSize:22,fontWeight:700,letterSpacing:"-.02em",marginBottom:4}}>Models Configuration</h1>
          <p style={{fontSize:13,color:"var(--dim)",marginTop:3,marginBottom:18}}>
            Sales behavior models {"\u2014"} basket size, pricing, brand popularity, and returns. Changes are in-memory only.
          </p>

          {/* Tab bar */}
          <div style={{display:"flex",gap:0,marginBottom:18,borderBottom:"2px solid var(--border)"}}>
            {[["visual","Visual"],["yaml","YAML"]].map(([k,label])=>(
              <button key={k} onClick={()=>setModelsTab(k)} style={{padding:"9px 22px",fontSize:13,fontWeight:modelsTab===k?600:400,color:modelsTab===k?"var(--accent)":"var(--dim)",background:"none",border:"none",borderBottom:modelsTab===k?"2px solid var(--accent)":"2px solid transparent",marginBottom:-2,cursor:"pointer",fontFamily:"var(--sans)",transition:"all .12s"}}>{label}</button>
            ))}
          </div>

          {modelsTab==="yaml"?(
            <YamlEditor value={modelsYaml} onChange={setModelsYaml} filename="models.yaml" dirty={modelsDirty} applied={modelsApplied} error={modelsErr} onApply={saveModels} onReset={resetModels} />
          ):mf?(
            <div>
            {/* ── Macro Demand ── */}
            <Section num="M" title="Macro Demand" defaultOpen={true}>
              <R2>
                <F label="Mode" help="'once' plays factors once, 'repeat' loops them."><Sel value={mf.demandMode} onChange={v=>sm("demandMode",v)} options={["once","repeat"]} /></F>
                <F label="Year-level factors" help="Comma-separated demand multipliers per year.">
                  <input type="text" style={iS} value={(mf.demandFactors||[]).join(", ")} onChange={e=>sm("demandFactors",e.target.value.split(",").map(x=>parseFloat(x.trim())).filter(x=>!isNaN(x)))} />
                </F>
              </R2>
              {mf.demandFactors&&mf.demandFactors.length>0&&<div style={{marginTop:10,display:"flex",alignItems:"flex-end",gap:2,height:60}}>
                {mf.demandFactors.map((f,i)=>{const max=Math.max(...mf.demandFactors);return(
                  <div key={i} style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",gap:2}}>
                    <div style={{width:"100%",background:"var(--accent)",borderRadius:"3px 3px 0 0",height:Math.max(4,f/max*48),opacity:.7+.3*(f/max),transition:"height .2s"}} />
                    <span style={{fontSize:9,color:"var(--muted)",fontFamily:"var(--mono)"}}>{f.toFixed(1)}</span>
                  </div>
                );})}
              </div>}
            </Section>

            {/* ── Quantity (basket size) ── */}
            <Section num="Q" title="Quantity (Basket Size)">
              <R3>
                <F label="Poisson lambda" help="Base expected items per order."><N value={mf.qtyLambda} onChange={v=>sm("qtyLambda",v)} min={0.1} max={10} step={.1} /></F>
                <F label="Min qty"><N value={mf.qtyMin} onChange={v=>sm("qtyMin",v)} min={1} step={1} /></F>
                <F label="Max qty"><N value={mf.qtyMax} onChange={v=>sm("qtyMax",v)} min={1} step={1} /></F>
              </R3>
              <Sld label="Noise sigma" value={mf.qtyNoise} min={0} max={.5} step={.01} onChange={v=>sm("qtyNoise",v)} />
              <Box title="Monthly seasonality factors">
                <div style={{display:"grid",gridTemplateColumns:"repeat(12,1fr)",gap:6}}>
                  {MONTHS.map((mon,i)=>(
                    <div key={mon} style={{textAlign:"center"}}>
                      <div style={{fontSize:10,color:"var(--muted)",marginBottom:3}}>{mon}</div>
                      <input type="number" style={{...iS,padding:"5px 3px",textAlign:"center",fontSize:12}} value={(mf.qtyMonthly||[])[i]||1} min={.5} max={2} step={.01}
                        onChange={e=>{const arr=[...(mf.qtyMonthly||Array(12).fill(1))];arr[i]=parseFloat(e.target.value)||1;sm("qtyMonthly",arr);}} />
                    </div>
                  ))}
                </div>
              </Box>
            </Section>

            {/* ── Pricing ── */}
            <Section num="P" title="Pricing" defaultOpen={false}>
              <Box title="Inflation">
                <R3>
                  <F label="Annual rate"><N value={mf.inflationRate} onChange={v=>sm("inflationRate",v)} min={0} max={.5} step={.005} /></F>
                  <F label="Month volatility"><N value={mf.inflationVolatility} onChange={v=>sm("inflationVolatility",v)} min={0} max={.1} step={.001} /></F>
                  <F label="Factor clip range">
                    <div style={{display:"flex",gap:6,alignItems:"center"}}>
                      <N value={mf.inflationClipMin} onChange={v=>sm("inflationClipMin",v)} min={.5} max={2} step={.01} style={{width:80}} />
                      <span style={{color:"var(--muted)"}}>–</span>
                      <N value={mf.inflationClipMax} onChange={v=>sm("inflationClipMax",v)} min={.5} max={3} step={.01} style={{width:80}} />
                    </div>
                  </F>
                </R3>
              </Box>
              <Box title="Markdown">
                <Check checked={mf.markdownEnabled} onChange={v=>sm("markdownEnabled",v)} label="Enable markdowns" />
                {mf.markdownEnabled&&<div style={{marginTop:10}}><R3>
                  <F label="Max % of price"><N value={mf.markdownMaxPct} onChange={v=>sm("markdownMaxPct",v)} min={0} max={1} step={.05} /></F>
                  <F label="Min net price"><N value={mf.markdownMinNet} onChange={v=>sm("markdownMinNet",v)} min={0} step={.01} /></F>
                  <F label=" "><div style={{paddingTop:16}}><Check checked={mf.markdownAllowNeg} onChange={v=>sm("markdownAllowNeg",v)} label="Allow negative margin" /></div></F>
                </R3></div>}
                {mf.markdownEnabled&&mf.markdownLadder&&mf.markdownLadder.length>0&&<Box title="Discount ladder">
                  <div style={{display:"grid",gridTemplateColumns:"60px 80px 1fr 50px",gap:"4px 10px",alignItems:"center",fontSize:12}}>
                    <span style={{fontWeight:600,color:"var(--muted)",fontSize:10}}>KIND</span>
                    <span style={{fontWeight:600,color:"var(--muted)",fontSize:10}}>VALUE</span>
                    <span style={{fontWeight:600,color:"var(--muted)",fontSize:10}}>WEIGHT BAR</span>
                    <span style={{fontWeight:600,color:"var(--muted)",fontSize:10,textAlign:"right"}}>WT</span>
                    {mf.markdownLadder.map((step,i)=>(
                      <React.Fragment key={i}>
                        <span style={{fontFamily:"var(--mono)",color:"var(--dim)"}}>{step.kind}</span>
                        <span style={{fontFamily:"var(--mono)",color:"var(--text)"}}>{step.kind==="none"?"—":"$"+step.value}</span>
                        <div style={{height:14,background:"var(--border)",borderRadius:3,overflow:"hidden"}}><div style={{height:"100%",width:`${step.weight*100/.35*100}%`,maxWidth:"100%",background:"var(--accent)",borderRadius:3,opacity:.7}} /></div>
                        <span style={{fontFamily:"var(--mono)",color:"var(--accent)",textAlign:"right"}}>{(step.weight*100).toFixed(0)}%</span>
                      </React.Fragment>
                    ))}
                  </div>
                  <div style={{fontSize:10.5,color:"var(--muted)",marginTop:6}}>Edit ladder values in the YAML tab.</div>
                </Box>}
              </Box>
            </Section>

            {/* ── Brand Popularity ── */}
            <Section num="B" title="Brand Popularity" defaultOpen={false}>
              <Check checked={mf.brandEnabled} onChange={v=>sm("brandEnabled",v)} label="Enable brand popularity rotation" />
              {mf.brandEnabled&&<>
                <R2>
                  <F label="Winner boost" help="Multiplier for the 'winning' brand each year."><N value={mf.brandWinnerBoost} onChange={v=>sm("brandWinnerBoost",v)} min={1} max={10} step={.1} /></F>
                  <F label="Seed"><N value={mf.brandSeed} onChange={v=>sm("brandSeed",v)} min={0} step={1} /></F>
                </R2>
                {mf.brandWeights&&Object.keys(mf.brandWeights).length>0&&<Box title="Brand weight overrides">
                  {Object.entries(mf.brandWeights).map(([brand,w])=>(
                    <div key={brand} style={{display:"flex",alignItems:"center",gap:10,marginBottom:6}}>
                      <span style={{fontSize:13,width:160,color:"var(--text)"}}>{brand}</span>
                      <input type="range" min={0} max={1} step={.01} value={w} onChange={e=>{const nw={...mf.brandWeights,[brand]:parseFloat(e.target.value)};sm("brandWeights",nw);}} style={{flex:1,accentColor:"var(--accent)"}} />
                      <span style={{fontSize:12,fontFamily:"var(--mono)",color:"var(--accent)",width:42,textAlign:"right"}}>{(w*100).toFixed(0)}%</span>
                    </div>
                  ))}
                </Box>}
              </>}
            </Section>

            {/* ── Returns ── */}
            <Section num="R" title="Returns" defaultOpen={false}>
              <Check checked={mf.retEnabled} onChange={v=>sm("retEnabled",v)} label="Enable return modeling" />
              {mf.retEnabled&&<>
                <R3>
                  <F label="Lag distribution"><Sel value={mf.retLagDist} onChange={v=>sm("retLagDist",v)} options={["triangular","uniform","normal"]} /></F>
                  <F label="Lag mode (days)" help="Peak of the triangular distribution."><N value={mf.retLagMode} onChange={v=>sm("retLagMode",v)} min={1} max={90} step={1} /></F>
                  <F label="Full-line return %"><N value={mf.retFullLinePct} onChange={v=>sm("retFullLinePct",v)} min={0} max={1} step={.01} /></F>
                </R3>
                {mf.retReasons&&mf.retReasons.length>0&&<Box title="Return reasons">
                  <div style={{display:"grid",gridTemplateColumns:"30px 1fr 1fr 50px",gap:"4px 10px",alignItems:"center",fontSize:12}}>
                    <span style={{fontWeight:600,color:"var(--muted)",fontSize:10}}>KEY</span>
                    <span style={{fontWeight:600,color:"var(--muted)",fontSize:10}}>REASON</span>
                    <span style={{fontWeight:600,color:"var(--muted)",fontSize:10}}>WEIGHT</span>
                    <span style={{fontWeight:600,color:"var(--muted)",fontSize:10,textAlign:"right"}}>%</span>
                    {mf.retReasons.map((r,i)=>(
                      <React.Fragment key={i}>
                        <span style={{fontFamily:"var(--mono)",color:"var(--muted)"}}>{r.key}</span>
                        <span style={{color:"var(--text)"}}>{r.label}</span>
                        <div style={{height:14,background:"var(--border)",borderRadius:3,overflow:"hidden"}}><div style={{height:"100%",width:`${r.weight*100/.28*100}%`,maxWidth:"100%",background:"var(--accent)",borderRadius:3,opacity:.7}} /></div>
                        <span style={{fontFamily:"var(--mono)",color:"var(--accent)",textAlign:"right"}}>{(r.weight*100).toFixed(0)}%</span>
                      </React.Fragment>
                    ))}
                  </div>
                  <div style={{fontSize:10.5,color:"var(--muted)",marginTop:6}}>Edit reasons in the YAML tab.</div>
                </Box>}
              </>}
            </Section>
            </div>
          ):(<div style={{padding:40,textAlign:"center",color:"var(--muted)"}}>Loading models...</div>)}
        </div>

        ):page==="config"?(
        /* ═══ CONFIG YAML EDITOR PAGE ═══ */
        <div>
          <button onClick={()=>setPage("main")} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"6px 14px",borderRadius:7,border:"1px solid var(--border)",background:"var(--surface)",color:"var(--dim)",fontSize:12.5,cursor:"pointer",fontFamily:"var(--sans)",marginBottom:18,transition:"all .12s"}} onMouseOver={e=>e.currentTarget.style.borderColor="var(--accent)"} onMouseOut={e=>e.currentTarget.style.borderColor="var(--border)"}>
            {"\u2190"} Back to configuration
          </button>
          <h1 style={{fontSize:22,fontWeight:700,letterSpacing:"-.02em",marginBottom:6}}>Config YAML</h1>
          <p style={{fontSize:13,color:"var(--dim)",marginTop:0,marginBottom:16,lineHeight:1.5}}>
            Edit the full config below. Applying here will override any preset or form settings. The file on disk is never modified.
          </p>
          {cfgYamlDirty&&<div style={{padding:"9px 14px",marginBottom:14,borderRadius:8,background:"var(--warnBg)",border:"1px solid rgba(202,138,4,.1)",fontSize:12,color:"var(--warn)",display:"flex",alignItems:"center",gap:8}}>
            <span style={{width:7,height:7,borderRadius:"50%",background:"var(--warn)",flexShrink:0}} />
            You have unsaved edits. Press Apply (Ctrl+S) to update in-memory config.
          </div>}
          <YamlEditor value={cfgYaml} onChange={setCfgYaml} filename="config.yaml" dirty={cfgYamlDirty} error={cfgYamlErr} onApply={saveCfgYaml} onReset={resetCfgYaml} onRefresh={refreshCfgYamlFromUI} />
        </div>

        ):(
        /* ═══ MAIN CONFIG PAGE ═══ */
        <div>
        <h1 style={{fontSize:22,fontWeight:700,letterSpacing:"-.02em"}}>Retail Data Generator</h1>
        <p style={{fontSize:13,color:"var(--dim)",marginTop:3,marginBottom:22}}>Configure and generate large, realistic retail datasets</p>

        {renderConfigForm()}

        {/* 19 GENERATE */}
        <Section num="6" title="Generate" defaultOpen={true}>
          <div style={{marginTop:10,padding:"12px 16px",background:"var(--alt)",borderRadius:10,border:"1px solid var(--border)",fontSize:13,color:"var(--dim)",lineHeight:1.7}}>
            <strong style={{color:"var(--text)"}}>{cfg.salesRows.toLocaleString()}</strong> rows · <strong style={{color:"var(--text)"}}>{cfg.customers.toLocaleString()}</strong> cust · <strong style={{color:"var(--text)"}}>{cfg.products.toLocaleString()}</strong> prod · {cfg.startDate} {"\u2192"} {cfg.endDate} · <Badge>{cfg.format.toUpperCase()}</Badge>
            {cfg.salesOutput!=="sales"&&<>{" · "}<Badge>{cfg.salesOutput==="both"?"Sales + Orders":"Orders"}</Badge></>}
            {cfg.returnsEnabled&&<>{" · "}<Badge variant="success">Returns {(cfg.returnRate*100).toFixed(1)}%</Badge></>}
            {cfg.budgetEnabled&&<>{" · "}<Badge>Budget</Badge></>}
            {cfg.inventoryEnabled&&<>{" · "}<Badge>Inventory</Badge></>}
            {cfg.csEnabled&&<>{" · "}<Badge>Segments</Badge></>}
            {cfg.spEnabled&&<>{" · "}<Badge>Superpowers</Badge></>}
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

      {/* ═══ STICKY GENERATE BAR (main page only) ═══ */}
      {page==="main"&&(
        <div className="sticky-generate-bar" style={{position:"fixed",bottom:0,right:0,left:sidebarW,background:"var(--surface)",borderTop:"1px solid var(--border)",padding:"10px 24px",zIndex:90,display:"flex",alignItems:"center",gap:14,boxShadow:"0 -2px 12px var(--shadow)",transition:"left .2s"}}>
          <div style={{flex:1,display:"flex",alignItems:"center",gap:12,fontSize:12,color:"var(--dim)",overflow:"hidden"}}>
            <strong style={{color:"var(--text)",flexShrink:0}}>{cfg.salesRows.toLocaleString()} rows</strong>
            <span style={{color:"var(--muted)"}}>{cfg.startDate} {"\u2192"} {cfg.endDate}</span>
            {isRunning&&<><span style={{width:7,height:7,borderRadius:"50%",background:"var(--ok)",animation:"pulse 1.5s infinite",flexShrink:0}} /><span style={{color:"var(--ok)",fontFamily:"var(--mono)"}}>{elapsed<60?elapsed.toFixed(1)+"s":Math.floor(elapsed/60)+"m "+Math.floor(elapsed%60)+"s"}</span></>}
            {errors.length>0&&<Badge variant="error">{errors.length} error{errors.length>1?"s":""}</Badge>}
            <span style={{fontSize:10.5,color:"var(--muted)",marginLeft:"auto",flexShrink:0}}>Ctrl+Enter</span>
          </div>
          {isRunning?
            <button onClick={cancelGenerate} style={{padding:"8px 20px",borderRadius:8,border:"1px solid var(--err)",background:"var(--errBg)",color:"var(--err)",fontWeight:600,fontSize:12.5,cursor:"pointer",fontFamily:"var(--sans)",whiteSpace:"nowrap"}}>Cancel</button>
          :
            <button onClick={runGenerate} disabled={errors.length>0} style={{padding:"8px 24px",borderRadius:8,border:"none",fontSize:13,fontWeight:700,cursor:errors.length>0?"not-allowed":"pointer",fontFamily:"var(--sans)",background:errors.length>0?"var(--alt)":"var(--accent)",color:errors.length>0?"var(--muted)":"#fff",whiteSpace:"nowrap",boxShadow:errors.length===0?"0 2px 8px rgba(79,91,213,.2)":"none"}}>Generate</button>
          }
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
