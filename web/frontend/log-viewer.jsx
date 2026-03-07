/* ═══ log-viewer.jsx — pipeline log output ═══ */

function LogViewer({logs,isRunning,elapsed}){
  const ref=useRef(null);
  useEffect(()=>{if(ref.current)ref.current.scrollTop=ref.current.scrollHeight;},[logs]);

  const cls=l=>{
    if(l.includes("completed")||l.includes("written")||l.includes("success"))return"var(--ok)";
    if(l.includes("Error")||l.includes("FAIL")||l.includes("failed"))return"var(--err)";
    if(l.startsWith("  "))return"#4f5bd5";
    return"var(--text)";
  };
  const fmt=s=>s<60?`${s.toFixed(1)}s`:`${Math.floor(s/60)}m ${Math.floor(s%60)}s`;

  return(
    <div>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginTop:12}}>
        <div style={{display:"flex",alignItems:"center",gap:7}}>
          <span style={{width:8,height:8,borderRadius:"50%",background:isRunning?"var(--ok)":"var(--muted)",animation:isRunning?"pulse 1.5s infinite":"none"}} />
          <span style={{fontSize:12,color:"var(--dim)"}}>{isRunning?"Running...":logs.length>0?"Completed":"Waiting"}</span>
        </div>
        {elapsed>0&&<span style={{fontSize:12,color:"var(--accent)",fontFamily:"var(--mono)"}}>{fmt(elapsed)}</span>}
      </div>
      <div ref={ref} style={{background:"var(--surface)",border:"1px solid var(--border)",borderRadius:10,padding:"14px 16px",fontFamily:"var(--mono)",fontSize:12,lineHeight:1.7,height:360,overflowY:"auto",marginTop:10,whiteSpace:"pre-wrap"}}>
        {logs.length===0
          ?<span style={{color:"var(--muted)"}}>Pipeline output will appear here...</span>
          :logs.map((l,i)=><div key={i} style={{color:cls(l)}}>{l}</div>)}
      </div>
    </div>
  );
}
