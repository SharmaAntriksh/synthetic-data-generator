/* ═══ log-viewer.jsx — pipeline log output with collapsible stages ═══ */

function LogViewer({logs,isRunning,elapsed}){
  const ref=useRef(null);
  const[collapsed,setCollapsed]=useState({});
  useEffect(()=>{if(ref.current)ref.current.scrollTop=ref.current.scrollHeight;},[logs]);

  const cls=l=>{
    if(l.includes("completed")||l.includes("written")||l.includes("success"))return"var(--ok)";
    if(l.includes("Error")||l.includes("FAIL")||l.includes("failed"))return"var(--err)";
    if(l.startsWith("  "))return"var(--accent)";
    return"var(--text)";
  };
  const fmt=s=>s<60?`${s.toFixed(1)}s`:`${Math.floor(s/60)}m ${Math.floor(s%60)}s`;

  /* Parse logs into stages */
  const parsed=useMemo(()=>{
    const stages=[];
    let current=null;
    /* Stage markers: lines like "═══ Stage Name ═══" or "--- Stage Name ---" or lines ending with "..." that start a group */
    const stageRe=/^[═\-\s]*([\w\s]+?)[═\-\s]*$/;
    const sectionRe=/^(Generating|Building|Loading|Writing|Processing|Computing|Merging|Running|Creating|Validating)\s/i;

    for(let i=0;i<logs.length;i++){
      const l=logs[i];
      const isStage=stageRe.test(l)&&l.includes("═");
      const isSection=!l.startsWith("  ")&&sectionRe.test(l.trim());

      if(isStage||isSection){
        const name=isStage?l.replace(/[═\-]/g,"").trim():l.trim();
        current={name,lines:[],index:i};
        stages.push(current);
      }else if(current){
        current.lines.push({text:l,index:i});
      }else{
        stages.push({name:null,lines:[{text:l,index:i}],index:i});
      }
    }
    return stages;
  },[logs]);

  const toggleStage=(idx)=>setCollapsed(p=>({...p,[idx]:!p[idx]}));

  /* Progress: count completed stages vs total */
  const totalStages=parsed.filter(s=>s.name).length;
  const doneStages=parsed.filter(s=>s.name&&s.lines.some(l=>l.text.includes("completed")||l.text.includes("written")||l.text.includes("success")||l.text.includes("done"))).length;
  const pct=totalStages>0?Math.round(doneStages/totalStages*100):0;

  return(
    <div>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:12}}>
        <div style={{display:"flex",alignItems:"center",gap:7}}>
          <span style={{width:8,height:8,borderRadius:"50%",background:isRunning?"var(--ok)":"var(--muted)",animation:isRunning?"pulse 1.5s infinite":"none"}} />
          <span style={{fontSize:12,color:"var(--dim)"}}>{isRunning?"Running...":logs.length>0?"Completed":"Waiting"}</span>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          {isRunning&&totalStages>0&&<span style={{fontSize:11,color:"var(--muted)",fontFamily:"var(--mono)"}}>{doneStages}/{totalStages} stages</span>}
          {elapsed>0&&<span style={{fontSize:12,color:"var(--accent)",fontFamily:"var(--mono)"}}>{fmt(elapsed)}</span>}
        </div>
      </div>

      {/* Progress bar */}
      {(isRunning||logs.length>0)&&totalStages>0&&(
        <div style={{height:3,background:"var(--border)",borderRadius:2,marginTop:8,overflow:"hidden"}}>
          <div style={{height:"100%",width:`${isRunning?pct:100}%`,background:isRunning?"var(--accent)":"var(--ok)",borderRadius:2,transition:"width .3s"}} />
        </div>
      )}

      <div ref={ref} style={{background:"var(--surface)",border:"1px solid var(--border)",borderRadius:10,padding:"10px 0",fontFamily:"var(--mono)",fontSize:12,lineHeight:1.7,height:360,overflowY:"auto",marginTop:10}}>
        {logs.length===0
          ?<span style={{color:"var(--muted)",padding:"4px 16px",display:"block"}}>Pipeline output will appear here...</span>
          :parsed.map((stage,si)=>{
            if(!stage.name){
              return stage.lines.map((l,li)=><div key={`${si}-${li}`} style={{color:cls(l.text),padding:"0 16px"}}>{l.text}</div>);
            }
            const isCollapsed=collapsed[si];
            return(
              <div key={si}>
                <div className="log-stage" onClick={()=>toggleStage(si)} style={{padding:"4px 16px",cursor:"pointer",display:"flex",alignItems:"center",gap:6,fontWeight:600,fontSize:12,color:"var(--accent)",userSelect:"none"}}>
                  <span className={`log-stage-arrow${isCollapsed?" collapsed":""}`} style={{fontSize:10,transition:"transform .15s",transform:isCollapsed?"rotate(-90deg)":"rotate(0)"}}>{"▾"}</span>
                  {stage.name}
                  <span style={{fontSize:10,color:"var(--muted)",fontWeight:400,marginLeft:4}}>({stage.lines.length} lines)</span>
                </div>
                {!isCollapsed&&stage.lines.map((l,li)=><div key={li} style={{color:cls(l.text),padding:"0 16px 0 32px"}}>{l.text}</div>)}
              </div>
            );
          })}
      </div>
    </div>
  );
}
