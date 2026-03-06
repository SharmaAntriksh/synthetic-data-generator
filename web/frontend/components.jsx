/* ═══ components.jsx — shared UI primitives ═══ */
const {useState,useEffect,useRef,useCallback}=React;

const API="/api";
const MONTHS=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const DAYS=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
const DIMS=["customers","products","stores","geography","promotions","dates","currency","exchange_rates"];

/* Shared input style */
const iS={width:"100%",background:"var(--bg)",border:"1px solid var(--border)",borderRadius:8,padding:"8px 11px",fontSize:13.5,color:"var(--text)",outline:"none",fontFamily:"var(--mono)"};

function Section({num,title,children,defaultOpen=true,badge}){
  const[open,setOpen]=useState(defaultOpen);
  return(
    <div style={{background:"var(--surface)",border:"1px solid var(--border)",borderRadius:10,marginBottom:14,boxShadow:"0 1px 3px rgba(0,0,0,.04)",overflow:"hidden"}}>
      <button onClick={()=>setOpen(!open)} style={{display:"flex",alignItems:"center",gap:10,padding:"14px 18px",cursor:"pointer",background:"none",border:"none",width:"100%",color:"var(--text)",fontFamily:"var(--sans)"}}>
        <span style={{width:26,height:26,borderRadius:7,display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,fontWeight:700,fontFamily:"var(--mono)",background:open?"var(--accent)":"var(--alt)",color:open?"#fff":"var(--muted)",transition:"all .15s"}}>{num}</span>
        <span style={{fontSize:14.5,fontWeight:600,flex:1,textAlign:"left"}}>{title}</span>
        {badge&&<Badge variant={badge.v}>{badge.t}</Badge>}
        <span style={{fontSize:16,color:"var(--muted)",transform:open?"rotate(0)":"rotate(-90deg)",transition:"transform .2s"}}>▾</span>
      </button>
      {open&&<div style={{padding:"0 18px 18px",borderTop:"1px solid var(--border)"}}>{children}</div>}
    </div>
  );
}

function F({label,help,children}){
  return(
    <div style={{marginTop:14}}>
      {label&&<label style={{display:"block",fontSize:12.5,fontWeight:500,color:"var(--dim)",marginBottom:5}}>{label}</label>}
      {children}
      {help&&<div style={{fontSize:11,color:"var(--muted)",marginTop:3}}>{help}</div>}
    </div>
  );
}

function N({value,onChange,min,max,step,style:sx}){
  const v=value!=null?value:"";
  return <input type="number" style={{...iS,...sx}} value={v} min={min} max={max} step={step||1}
    onChange={e=>{const n=parseFloat(e.target.value);onChange(isNaN(n)?(min||0):n);}}
    onFocus={e=>{e.target.style.borderColor="var(--focus)";e.target.style.boxShadow="0 0 0 3px var(--glow)";}}
    onBlur={e=>{e.target.style.borderColor="var(--border)";e.target.style.boxShadow="none";}} />;
}

function Sel({value,onChange,options,labels}){
  return <select style={{...iS,fontFamily:"var(--sans)",cursor:"pointer"}} value={value} onChange={e=>onChange(e.target.value)}>
    {options.map((o,i)=><option key={o} value={o}>{labels?labels[i]:o}</option>)}
  </select>;
}

function Check({checked,onChange,label,disabled}){
  return(
    <label onClick={e=>{e.preventDefault();if(!disabled)onChange(!checked);}}
      style={{display:"flex",alignItems:"center",gap:7,cursor:disabled?"default":"pointer",fontSize:13,color:disabled?"var(--muted)":"var(--text)",fontFamily:"var(--sans)",opacity:disabled?.6:1}}>
      <span style={{width:17,height:17,borderRadius:5,flexShrink:0,border:`2px solid ${checked?"var(--accent)":"var(--border)"}`,background:checked?"var(--accent)":"transparent",display:"flex",alignItems:"center",justifyContent:"center",transition:"all .12s"}}>
        {checked&&<span style={{color:"#fff",fontSize:10,fontWeight:700}}>✓</span>}
      </span>
      {label}
    </label>
  );
}

function Sld({label,value,onChange,min,max,step,fmt}){
  const v=Number(value)||0;
  return(
    <div style={{marginTop:10}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
        <span style={{fontSize:12,color:"var(--dim)"}}>{label}</span>
        <span style={{fontSize:12,color:"var(--accent)",fontFamily:"var(--mono)",fontWeight:600}}>{fmt?fmt(v):v.toFixed(2)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={v}
        onChange={e=>onChange(parseFloat(e.target.value))} style={{width:"100%",accentColor:"var(--accent)"}} />
    </div>
  );
}

function Badge({variant="default",children}){
  const c={
    default:{bg:"var(--glow)",color:"var(--accent)"},
    success:{bg:"var(--okBg)",color:"var(--ok)"},
    warning:{bg:"var(--warnBg)",color:"var(--warn)"},
    error:{bg:"var(--errBg)",color:"var(--err)"},
  }[variant];
  return <span style={{display:"inline-flex",padding:"2px 9px",borderRadius:6,fontSize:11.5,fontWeight:600,fontFamily:"var(--mono)",background:c.bg,color:c.color}}>{children}</span>;
}

function R2({children}){return <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>{children}</div>;}
function R3({children}){return <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:14}}>{children}</div>;}
function R4({children}){return <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:12}}>{children}</div>;}

function Box({title,children}){
  return(
    <div style={{marginTop:16,padding:"14px 16px",background:"var(--alt)",borderRadius:10,border:"1px solid var(--border)"}}>
      {title&&<div style={{fontSize:12,fontWeight:600,color:"var(--dim)",marginBottom:10}}>{title}</div>}
      {children}
    </div>
  );
}
