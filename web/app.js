/* Godhood Trials — original Canvas renderer. World rules live on the server. */
"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("world"), ctx = canvas.getContext("2d", {alpha: false});
const mini = $("minimap"), miniCtx = mini.getContext("2d");
const terrainCanvas = document.createElement("canvas"), terrainCtx = terrainCanvas.getContext("2d");
const TILE = 32;
const camera = {x: 32.5 * TILE, y: 34.5 * TILE, zoom: 1.15, follow: true};
let state = null, terrain = null, selected = "player", pollBusy = false, terrainNeeded = true;
let lastRevision = -1, lastSeed = null, rosterBuilt = false, lastMessageSerial = null, journalKey = "";
let viewWidth = 800, viewHeight = 500, toastTimer, lastPollTime = 0;
const tracks = new Map(), pressed = new Set();
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
let drag = null, pointer = null;
let building = false, buildKind = "wall", hoverTile = null, senseView = false, paletteBuilt = false;
let visibleTiles = new Set(), memoryKey = "";
const directions = {north:[0,-1],east:[1,0],south:[0,1],west:[-1,0]};
const memoryColors = ["#142b29","#85965d","#699d99","#bbae7e","#a28759","#60744f","#456945","#8c5871","#91977f","#b59663","#cdc087","#756048","#cfbb97","#9d9272","#6e5940","#b5a26a"];
function sensory(){return state?.perception?.resident===selected?state.perception:null;}
function canSee(x,y){return !senseView||visibleTiles.has(`${x},${y}`);}

function rect(c, color, x, y, w, h) { c.fillStyle = color; c.fillRect(Math.round(x), Math.round(y), w, h); }
function hash(x, y, n = 0) { const v = Math.sin(x * 127.1 + y * 311.7 + n * 73.7) * 43758.5453; return v - Math.floor(v); }
function shade(hex, amount) { const n = parseInt(hex.slice(1), 16); return `rgb(${[16,8,0].map(s => Math.max(0,Math.min(255,((n>>s)&255)+amount))).join(",")})`; }

function drawCreature(c, x, y, a, phase = 0, scale = 1, facing = a.facing) {
  c.save(); c.translate(Math.round(x), Math.round(y)); c.scale(scale, scale); if(a.unconscious)c.rotate(-Math.PI/2);
  const walk = a.action === "Walking" && !reducedMotion ? Math.round(Math.sin(phase) * 2) : 0;
  c.fillStyle = "#17352555"; c.beginPath(); c.ellipse(0, 0, 10, 3, 0, 0, Math.PI*2); c.fill();
  const edge = "#354c3c", base = a.color, light = shade(base, 17), dark = shade(base, -24);
  rect(c,edge,-8,-17,16,15); rect(c,edge,-10,-22,20,13);
  if(a.variant === 0){rect(c,edge,-10,-28,6,10);rect(c,edge,4,-28,6,10);rect(c,base,-9,-27,4,9);rect(c,base,5,-27,4,9);rect(c,dark,-8,-26,2,5);rect(c,dark,6,-26,2,5);}
  else if(a.variant === 1){rect(c,edge,-13,-22,7,7);rect(c,edge,6,-22,7,7);rect(c,base,-12,-21,6,5);rect(c,base,6,-21,6,5);}
  else{rect(c,edge,-9,-27,5,7);rect(c,edge,4,-27,5,7);rect(c,base,-8,-26,3,7);rect(c,base,5,-26,3,7);rect(c,light,-3,-25,6,4);}
  rect(c,base,-7,-16,14,13);rect(c,light,-4,-14,8,8);rect(c,base,-9,-22,18,12);rect(c,light,-7,-22,13,3);
  rect(c,edge,-7,-3+walk,5,4);rect(c,edge,2,-3-walk,5,4);rect(c,dark,-6,-3+walk,4,3);rect(c,dark,2,-3-walk,4,3);
  rect(c,dark,-10,-12-walk,3,7);rect(c,dark,7,-12+walk,3,7);
  if(facing !== "north"){
    const shift = facing === "east" ? 2 : facing === "west" ? -2 : 0;
    rect(c,"#fff3d3",-6+shift,-19,4,4);rect(c,"#fff3d3",2+shift,-19,4,4);
    rect(c,"#293e33",-4+shift,-18,2,3);rect(c,"#293e33",3+shift,-18,2,3);
    rect(c,dark,-1+shift,-12,3,1);
  }else{rect(c,dark,-4,-20,2,2);rect(c,dark,3,-19,2,2);}
  if(Object.values(a.inventory || {}).reduce((s,n)=>s+n,0)>0){rect(c,"#4c4932",7,-13,6,9);rect(c,"#a18a56",8,-12,4,7);rect(c,"#c0a269",8,-12,4,2);}
  c.restore();
}

function drawTree(c,x,y,r,variant){
  c.fillStyle="#214d2c35";c.beginPath();c.ellipse(x+3,y+3,19,7,0,0,Math.PI*2);c.fill();
  rect(c,"#5e5638",x-4,y-15,8,18);rect(c,"#8a7950",x-2,y-15,3,17);rect(c,"#554d33",x-7,y,14,3);
  if(r.amount<=0){rect(c,"#b49a66",x-4,y-5,8,3);return;}
  const palette=variant>.5?["#386346","#4f8150","#699657","#8fa962"]:["#385d43","#567846","#739653","#93ad69"];
  rect(c,palette[0],x-15,y-33,31,22);rect(c,palette[0],x-20,y-27,40,12);
  rect(c,palette[1],x-17,y-35,33,22);rect(c,palette[1],x-11,y-42,23,14);
  rect(c,palette[2],x-13,y-35,22,13);rect(c,palette[2],x-7,y-42,13,14);rect(c,palette[2],x-17,y-27,8,9);
  rect(c,palette[3],x-8,y-39,9,3);rect(c,palette[3],x-13,y-32,8,3);rect(c,palette[2],x+11,y-24,4,7);
}

function drawResource(c,x,y,r,variant){
  if(r.kind==="crop"){
    const stage=r.stage??Math.min(2,Math.floor(((state?.tick??0)-r.planted)*3/480));
    rect(c,"#73583e",x-10,y-3,20,7);rect(c,"#779a50",x-1,y-7-stage*3,3,10+stage*3);
    rect(c,"#a1b76b",x-6,y-7-stage*3,6,3);rect(c,"#88a95b",x+2,y-5-stage*3,6,3);return;
  }
  if(r.kind==="thorns"){
    rect(c,"#344a3444",x-13,y-2,27,8);rect(c,"#596146",x-12,y-9,24,11);
    for(const [dx,dy]of [[-10,-6],[-5,-13],[2,-7],[7,-12]]){rect(c,"#6e604b",x+dx,y+dy,3,14);rect(c,"#d6c695",x+dx-3,y+dy+2,3,2);rect(c,"#d6c695",x+dx+3,y+dy+6,3,2);}
    return;
  }
  if(r.kind==="amber_bush"||r.kind==="amber_fruit"){
    if(r.kind==="amber_bush"){rect(c,"#526645",x-12,y-12,24,13);rect(c,"#7f8b4d",x-8,y-17,16,13);}
    const dots=r.kind==="amber_fruit"?[[0,-5]]:[[-7,-8],[2,-12],[6,-5],[-2,-3]];
    dots.slice(0,Math.min(r.amount,4)).forEach(([dx,dy])=>{rect(c,"#aa6536",x+dx,y+dy,5,5);rect(c,"#edb95d",x+dx,y+dy,3,3);});return;
  }
  if(r.kind==="tree"){drawTree(c,x,y,r,variant);return;}
  if(r.kind==="berry"){
    rect(c,"#467040",x-12,y-12,24,13);rect(c,"#669044",x-10,y-16,19,13);rect(c,"#8aa356",x-8,y-16,10,4);
    const dots=[[-7,-8],[2,-12],[6,-6],[-2,-5],[8,-11]];
    dots.slice(0,Math.min(r.amount,5)).forEach(([dx,dy])=>{rect(c,"#794258",x+dx,y+dy,4,4);rect(c,"#c88385",x+dx,y+dy,2,2);});
  }else if(r.kind==="stone"){
    rect(c,"#596a56",x-10,y-4,22,7);rect(c,"#89927a",x-11,y-10,18,11);rect(c,"#a3a58b",x-7,y-15,13,8);rect(c,"#bbb8a0",x-5,y-15,8,3);rect(c,"#6b7a67",x+6,y-7,7,8);
    if(r.amount===0)rect(c,"#62715a",x-7,y-7,8,3);
  }else{
    rect(c,"#36563044",x-6,y,14,4);
    if(r.kind==="wood"){rect(c,"#715636",x-8,y-5,17,6);rect(c,"#ba9761",x-8,y-5,3,6);rect(c,"#9c7c4b",x-4,y-5,13,2);}
    if(r.kind==="food"){rect(c,"#89516a",x-5,y-7,6,6);rect(c,"#a46778",x,y-5,6,6);rect(c,"#bda16a",x-2,y-9,2,4);}
    if(r.kind==="seed"){rect(c,"#d8c486",x-3,y-5,3,4);rect(c,"#c6b075",x+2,y-3,3,3);}
  }
}

function drawShelter(c,x,y){
  rect(c,"#314f3540",x-20,y-2,43,9);rect(c,"#6d6544",x-14,y-16,4,20);rect(c,"#6d6544",x+10,y-16,4,20);
  rect(c,"#676b3e",x-20,y-23,40,7);rect(c,"#929052",x-16,y-28,32,6);rect(c,"#b3a866",x-12,y-33,24,6);rect(c,"#ccbc7d",x-7,y-37,14,5);
  rect(c,"#d8c58a",x-14,y-27,3,6);rect(c,"#b3a469",x-19,y-21,36,2);rect(c,"#4a5138",x-10,y-16,20,2);
}

function drawStructure(c,x,y,s,tx,ty){
  if(s.kind==="shelter"){drawShelter(c,x,y);return;}
  if(s.kind==="floor")return;
  const joins=(dx,dy)=>["wall","stone_wall","door"].includes(state.structures[`${tx+dx},${ty+dy}`]?.kind);
  if(s.kind==="door"){
    rect(c,"#4d4932",x-14,y-26,4,29);rect(c,"#4d4932",x+10,y-26,4,29);rect(c,"#b09360",x-14,y-28,28,5);
    if(s.open){rect(c,"#977449",x-13,y-21,5,23);rect(c,"#c1a268",x-12,y-21,2,22);}
    else{rect(c,"#745638",x-9,y-22,18,25);for(let i=-7;i<9;i+=5)rect(c,"#a78351",x+i,y-21,3,22);rect(c,"#cfbd72",x+5,y-9,2,2);rect(c,"#5b4d33",x-9,y-4,18,3);}
    return;
  }
  const stone=s.kind==="stone_wall",left=joins(-1,0)?16:12,right=joins(1,0)?16:12;
  rect(c,"#253e3144",x-left,y-2,left+right+4,9);
  if(joins(0,1)){rect(c,stone?"#777f70":"#755735",x-7,y-4,14,21);rect(c,stone?"#a2a58d":"#b28e56",x-7,y-7,14,5);}
  if(joins(0,-1)){rect(c,stone?"#777f70":"#755735",x-7,y-41,14,19);rect(c,stone?"#a2a58d":"#b28e56",x-7,y-43,14,5);}
  rect(c,stone?"#747e6e":"#785c3c",x-left,y-23,left+right,27);
  if(stone){for(let row=0;row<3;row++)for(let i=-left;i<right;i+=11){rect(c,row%2?"#909884":"#858e7b",x+i+1,y-21+row*8,9,6);}rect(c,"#b5b69c",x-left,y-25,left+right,4);}
  else{for(let i=-left;i<right;i+=7){rect(c,"#a18252",x+i+1,y-23,5,27);rect(c,"#c0a16b",x+i+1,y-24,5,3);}rect(c,"#594d31",x-left,y-15,left+right,3);rect(c,"#594d31",x-left,y-4,left+right,3);}
}

function drawFloor(c,x,y,covered){
  rect(c,"#755f40",x,y,TILE,TILE);
  for(let row=0;row<4;row++){rect(c,covered?"#a18d63":"#b29a6c",x+1,y+row*8+1,30,6);rect(c,"#8a714d",x+(row%2?10:22),y+row*8+1,1,6);}
}

function placementCheck(x,y){
  const me=state.residents.find(a=>a.id==="player"),key=`${x},${y}`,s=state.structures[key];
  if(Math.abs(x-me.x)+Math.abs(y-me.y)>1)return "Move beside this tile.";
  if(buildKind==="remove")return s||(state.resources[key]?.amount===0&&state.resources[key]?.kind!=="crop")?"Remove and recover materials":"Nothing to remove here.";
  if(terrain[y]?.[x]!==0||state.resources[key]||(s&&(s.kind!=="floor"||buildKind==="floor")))return "Choose clear grass or a wooden floor.";
  if(["wall","stone_wall","door"].includes(buildKind)&&state.residents.some(a=>a.x===x&&a.y===y))return "This tile is occupied.";
  const cost=state.recipes?.[buildKind]?.cost||{};
  if(Object.entries(cost).some(([item,n])=>me.inventory[item]<n))return "Gather more materials.";
  return "Ready to place";
}

function buildTerrain(){
  terrainCanvas.width=state.size*TILE;terrainCanvas.height=state.size*TILE;
  terrainCtx.imageSmoothingEnabled=false;
  const grass=["#82935b","#87985f","#81925a","#8b9b62","#85985d"];
  for(let y=0;y<state.size;y++)for(let x=0;x<state.size;x++){
    const t=terrain[y][x], px=x*TILE,py=y*TILE,n=hash(x,y,state.seed);
    rect(terrainCtx,t===1?"#6b9e9a":t===2?"#b9af7c":t===3?"#8b7952":t===4?"#556d49":grass[Math.floor(n*grass.length)],px,py,TILE,TILE);
    if(t===0){
      for(let i=0;i<3;i++){const dx=Math.floor(hash(x,y,i+8)*28),dy=Math.floor(hash(x,y,i+20)*28);rect(terrainCtx,i===0?"#728651":"#99a66a",px+dx,py+dy,2,3);rect(terrainCtx,"#778c54",px+dx+3,py+dy+1,1,2);}
      if(n>.94){rect(terrainCtx,"#d3c482",px+8,py+17,2,2);rect(terrainCtx,"#c9bc75",px+20,py+8,2,2);}
    }else if(t===1){rect(terrainCtx,"#8bb8ac",px+4,py+10,12,2);rect(terrainCtx,"#60928f",px+18,py+25,10,2);}
    else if(t===2){rect(terrainCtx,"#c9bd89",px+3,py+4,12,2);rect(terrainCtx,"#a99d70",px+20,py+22,3,2);}
    else if(t===3){for(let d=0;d<32;d+=8){rect(terrainCtx,"#c0a16b",px+d+1,py+2,6,28);rect(terrainCtx,"#786343",px+d,py+2,1,28);}rect(terrainCtx,"#645939",px,py+3,32,3);rect(terrainCtx,"#645939",px,py+26,32,3);}
    else{rect(terrainCtx,"#61774f",px+4,py+4,23,20);rect(terrainCtx,"#768764",px+5,py+3,17,6);}
  }
}

function resize(){const bounds=canvas.getBoundingClientRect();viewWidth=bounds.width;viewHeight=bounds.height;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(viewWidth*dpr);canvas.height=Math.round(viewHeight*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.imageSmoothingEnabled=false;}
new ResizeObserver(resize).observe($("stage"));

function interpolated(a,time){const track=tracks.get(a.id);if(!track)return{x:a.x,y:a.y};const p=reducedMotion?1:Math.min(1,(time-track.time)/Math.min(220,track.duration));return{x:track.fromX+(a.x-track.fromX)*p,y:track.fromY+(a.y-track.fromY)*p};}
function render(time){
  requestAnimationFrame(render);if(!state||!terrain)return;
  const me=state.residents.find(a=>a.id==="player");
  const followed=state.residents.find(a=>a.id===selected)||me;
  if((camera.follow||senseView)&&followed){const p=senseView?followed:interpolated(followed,time);camera.x=(p.x+.5)*TILE;camera.y=(p.y+.5)*TILE;}
  ctx.fillStyle="#506d48";ctx.fillRect(0,0,viewWidth,viewHeight);
  ctx.save();ctx.translate(viewWidth/2,viewHeight/2);ctx.scale(camera.zoom,camera.zoom);ctx.translate(-Math.round(camera.x),-Math.round(camera.y));
  ctx.drawImage(terrainCanvas,0,0);
  const left=camera.x-viewWidth/2/camera.zoom-64,right=camera.x+viewWidth/2/camera.zoom+64,top=camera.y-viewHeight/2/camera.zoom-64,bottom=camera.y+viewHeight/2/camera.zoom+80;
  const entities=[];
  const covered=new Set(state.covered||[]);
  for(const [key,basket]of Object.entries(state.caches||{})){const [x,y]=key.split(",").map(Number);if(canSee(x,y)){rect(ctx,"#b08a55",x*TILE+5,y*TILE+9,22,15);rect(ctx,"#665037",x*TILE+7,y*TILE+11,18,4);if(basket.food)rect(ctx,"#bc7290",x*TILE+12,y*TILE+8,7,6);}}
  for(const [key,s]of Object.entries(state.structures)){const [x,y]=key.split(",").map(Number);if(!canSee(x,y))continue;if(s.kind==="floor"||s.floor)drawFloor(ctx,x*TILE,y*TILE,covered.has(key));entities.push({type:"structure",x:(x+.5)*TILE,y:(y+.75)*TILE,s,tx:x,ty:y});}
  for(const [key,r]of Object.entries(state.resources)){const [x,y]=key.split(",").map(Number);if(canSee(x,y)&&x*TILE>left&&x*TILE<right&&y*TILE>top&&y*TILE<bottom)entities.push({type:"resource",x:(x+.5)*TILE,y:(y+.75)*TILE,r,variant:hash(x,y)});}
  for(const a of state.residents){if(!canSee(a.x,a.y))continue;const p=senseView?a:interpolated(a,time);entities.push({type:"creature",x:(p.x+.5)*TILE,y:(p.y+.7)*TILE,a});}
  entities.sort((a,b)=>a.y-b.y);
  for(const e of entities){
    if(e.type==="resource")drawResource(ctx,e.x,e.y,e.r,e.variant);
    else if(e.type==="structure")drawStructure(ctx,e.x,e.y,e.s,e.tx,e.ty);
    else{
      if(e.a.id===selected){ctx.strokeStyle="#f0df9b";ctx.lineWidth=1.2;ctx.beginPath();ctx.ellipse(e.x,e.y+1,13,5,0,0,Math.PI*2);ctx.stroke();}
      drawCreature(ctx,e.x,e.y,e.a,state.paused?0:time/110);
      if(e.a.id==="player"&&!senseView){ctx.fillStyle="#f4e4ae";ctx.beginPath();ctx.moveTo(e.x,e.y-39);ctx.lineTo(e.x+3,e.y-35);ctx.lineTo(e.x,e.y-31);ctx.lineTo(e.x-3,e.y-35);ctx.closePath();ctx.fill();}
      if(camera.zoom>=.85||e.a.id===selected||(!senseView&&e.a.id==="player")){
        ctx.font="9px 'Segoe UI',sans-serif";ctx.textAlign="center";const width=ctx.measureText(e.a.name).width+10;rect(ctx,"#18362bd9",e.x-width/2,e.y+7,width,13);ctx.fillStyle="#eae5c9";ctx.fillText(e.a.name,e.x,e.y+17);
      }
    }
  }
  if(senseView){
    for(let y=Math.floor(top/TILE);y<=Math.ceil(bottom/TILE);y++)for(let x=Math.floor(left/TILE);x<=Math.ceil(right/TILE);x++)if(!canSee(x,y))rect(ctx,"#18312e",x*TILE,y*TILE,TILE,TILE);
  }
  if(building){
    for(const [dx,dy] of [[0,0],...Object.values(directions)]){const x=me.x+dx,y=me.y+dy;ctx.strokeStyle="#e1cf924f";ctx.lineWidth=1;ctx.strokeRect(x*TILE+1,y*TILE+1,TILE-2,TILE-2);}
    if(hoverTile&&canSee(hoverTile.x,hoverTile.y)){
      const {x,y}=hoverTile,ready=["Ready to place","Remove and recover materials"].includes(placementCheck(x,y));
      ctx.globalAlpha=.55;if(buildKind!=="remove"){if(buildKind==="floor")drawFloor(ctx,x*TILE,y*TILE,false);else drawStructure(ctx,(x+.5)*TILE,(y+.75)*TILE,{kind:buildKind},x,y);}ctx.globalAlpha=1;
      ctx.fillStyle=ready?"#b9d89333":"#df947544";ctx.fillRect(x*TILE,y*TILE,TILE,TILE);ctx.strokeStyle=ready?"#dbebae":"#f2a68a";ctx.lineWidth=2;ctx.strokeRect(x*TILE+1,y*TILE+1,TILE-2,TILE-2);
    }
  }
  ctx.restore();
  const hour=(6+state.dayProgress*24)%24;const darkness=hour>19?Math.min(.32,(hour-19)*.09):hour<6?.25:0;
  if(darkness){ctx.fillStyle=`rgba(15,36,47,${darkness})`;ctx.fillRect(0,0,viewWidth,viewHeight);}
  if(state.weather==="Rain"){
    ctx.fillStyle="#31585d24";ctx.fillRect(0,0,viewWidth,viewHeight);ctx.strokeStyle="#d1e2da65";ctx.lineWidth=1;ctx.beginPath();
    for(let i=0;i<45;i++){const x=hash(i,1)*viewWidth;const y=(hash(i,2)*viewHeight+(reducedMotion?0:time*.22))%viewHeight;ctx.moveTo(x,y);ctx.lineTo(x-3,y+12);}ctx.stroke();
  }
}
requestAnimationFrame(render);

function drawMinimap(){
  if(!state||!terrain)return;if(senseView){mini.hidden=true;return;}mini.hidden=false;miniCtx.imageSmoothingEnabled=false;miniCtx.drawImage(terrainCanvas,0,0,144,144);
  for(const key of Object.keys(state.structures)){const [x,y]=key.split(",").map(Number);rect(miniCtx,"#d9c49a",x/state.size*144,y/state.size*144,2,2);}
  for(const a of state.residents){rect(miniCtx,a.id==="player"?"#fff4c0":"#354f3a",a.x/state.size*144-1,a.y/state.size*144-1,3,3);}
  miniCtx.strokeStyle="#fff0b9a0";miniCtx.lineWidth=1;const factor=144/(state.size*TILE);miniCtx.strokeRect((camera.x-viewWidth/2/camera.zoom)*factor,(camera.y-viewHeight/2/camera.zoom)*factor,viewWidth/camera.zoom*factor,viewHeight/camera.zoom*factor);
}
setInterval(drawMinimap,350);

function updateCameraLabel(){
  const a=state?.residents.find(r=>r.id===selected);
  $("followButton").textContent=`◎ Follow ${selected==="player"?"you":a?.name||"character"}`;
  $("followButton").setAttribute("aria-pressed",String(camera.follow||senseView));
  $("cameraCaption").textContent=senseView?"INDIVIDUAL PERSPECTIVE":camera.follow?(selected==="player"?"YOUR TRAVELER":`FOLLOWING ${a?.name||"CHARACTER"}`):"OBSERVING THE VALLEY";
}
function selectResident(id,center=false){
  const a=state?.residents.find(r=>r.id===id);if(!a)return;
  selected=id;visibleTiles.clear();memoryKey="";updateInspector();
  if(center){camera.x=(a.x+.5)*TILE;camera.y=(a.y+.5)*TILE;camera.follow=true;camera.zoom=Math.max(1.15,camera.zoom);}
  updateCameraLabel();poll();
}
function showEveryone(){
  if(!state)return;setSenseView(false);camera.follow=false;
  const xs=state.residents.map(a=>(a.x+.5)*TILE),ys=state.residents.map(a=>(a.y+.5)*TILE);
  const left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
  camera.x=(left+right)/2;camera.y=(top+bottom)/2;
  camera.zoom=Math.min(1.65,Math.max(.15,Math.min(Math.max(80,viewWidth-110)/(right-left+TILE*4),Math.max(80,viewHeight-190)/(bottom-top+TILE*4))));
  updateCameraLabel();
}
function drawImpression(canvas,sketch){
  const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.fillStyle=memoryColors[0];c.fillRect(0,0,canvas.width,canvas.height);
  if(!sketch)return;
  for(let i=0;i<81;i++){c.fillStyle=memoryColors[parseInt(sketch[i],16)||0];c.fillRect((i%9)*canvas.width/9,Math.floor(i/9)*canvas.height/9,canvas.width/9,canvas.height/9);}
}
function updateSenses(){
  const perception=sensory();if(!perception){$("visionCount").textContent="Updating…";return;}
  const o=perception.observation,vm=o.visual_memory;
  $("visionCount").textContent=`${o.tiles.filter(t=>t.terrain>=0).length} visible tiles`;
  $("bodySense").textContent=(o.body.unconscious?"Passed out - ":"")+(o.body.sheltered?"Under cover · sheltered":o.body.rain?"Feeling rain · exposed":"Open air · dry")+(o.body.pain>0?` · pain ${Math.round(o.body.pain)}`:"");
  $("hungerSense").textContent=`${({satiated:"Comfortably fed",hungry:"Hungry: rest restores less energy",weak:"Weak with hunger: slower movement",starving:"Starving: prolonged hunger can cause fainting"})[o.body.hunger_stage]||""}${o.body.starvation_strain>0?` · strain ${Math.round(o.body.starvation_strain)}%`:""}`;
  const blocked=o.touch.filter(t=>t.blocked).map(t=>t.direction);
  $("touchSense").textContent=blocked.length?`Blocked: ${blocked.join(", ")}`:"Touch: clear on all sides";
  if(o.attention?.length)$("touchSense").textContent+=`; tapped from ${o.attention.at(-1).bearing}`;
  $("toneMemory").textContent=o.auditory_memory?.length?`Recent tones: ${o.auditory_memory.map(e=>e.tone).join(" > ")}`:"No recent tones";
  const learning=state.learning?.[selected];
  $("learningSense").textContent=learning?`${learning.backend==="laya-npu"?"Laya · local NPU · frozen weights":"Recurrent PPO · independent learner"}: ${learning.decisions} decisions / ${learning.updates} weight updates / ${Math.round(learning.nutrition)} nutrition`:"";
  let record=$("learningRecord");
  if(!record){record=document.createElement("details");record.id="learningRecord";$("learningSense").after(record);}
  record.hidden=!["recurrent-ppo","laya-npu"].includes(learning?.backend);
  if(!record.hidden){
    const wasOpen=record.open;record.replaceChildren();record.open=wasOpen;
    const title=document.createElement("summary");title.textContent="Learning record";record.append(title);
    const line=text=>{const p=document.createElement("p");p.textContent=text;record.append(p);};
    const actionLabel=a=>[a.verb,a.item,a.kind,a.direction,a.tone].filter(v=>v!==undefined&&v!==null&&v!=="").join(" ");
    if(learning.backend==="laya-npu"){
      line(`Laya chooses her own actions using local senses and up to ${learning.history_capacity} recent consequences. Her pretrained weights remain frozen.`);
      line(`Intel NPU · ${learning.observed_transitions} observed outcomes · latest decision used ${learning.diagnostics?.calls||0} model calls`);
      if(learning.diagnostics?.context_omissions?.length)line("Some older context was omitted to fit this model's input capacity.");
      for(const e of [...(learning.recent||[])].reverse().slice(0,3))line(`#${e.decision} ${actionLabel(e.action)} → observed fullness change ${e.observed.food.toFixed(1)}, health change ${e.observed.health.toFixed(1)}`);
      line("Compressed local senses and private context memory; no survival script or PPO model selects her actions.");
    }else{
    line(`Sequence learner · policy ${learning.policy_version} · ${learning.rollout_fill}/${learning.rollout_capacity} experiences before update`);
    line(`Curiosity reward ${learning.curiosity_coefficient>0?"experimental":"off"} · recent normalized prediction error ${learning.prediction_mae.toFixed(3)}`);
    if(learning.preferences?.length)line("Current choices: "+learning.preferences.map(e=>`${actionLabel(e.action)} ${Math.round(e.probability*100)}%`).join(" · "));
    for(const e of [...(learning.recent||[])].reverse().slice(0,3)){
      line(`#${e.decision} ${actionLabel(e.action)} → fullness: predicted ${e.predicted.fullness.toFixed(1)}, observed ${e.observed.fullness.toFixed(1)}; health: predicted ${e.predicted.health.toFixed(1)}, observed ${e.observed.health.toFixed(1)}`);
    }
    line("Measured predictions and outcomes from this resident's experience. Action probabilities are preferences, not confidence in success.");
    }
  }
  const soil=o.tiles.find(t=>t.dx===0&&t.dy===0)?.soil;
  $("ecologySense").textContent=o.ecology?.enabled?`Flask ${o.ecology.water_carried}/3${Number.isFinite(soil?.fertility)?` / soil ${Math.round(soil.fertility*100)}% / moisture ${Math.round(soil.moisture*100)}%`:""}${o.ecology.carried_cache!==null?` / carrying basket (${o.ecology.carried_cache.food} food)`:""}`:"";
  const sounds=[...new Set(o.hearing.map(s=>`${s.kind.replace("tone_", "tone ")} ${s.bearing}`))];
  $("hearingSense").textContent=sounds.length?`Hears ${sounds.slice(-2).join("; ")}`:"No nearby sounds";
  drawImpression($("senseImage"),vm?.current);
  $("memoryCount").textContent=`${vm?.count||0} / ${vm?.capacity||64} saved`;
  const key=JSON.stringify([selected,vm?.recalled]);if(key===memoryKey)return;memoryKey=key;
  $("memoryCards").replaceChildren();
  for(const memory of vm?.recalled||[]){
    const card=document.createElement("div");card.className="memory-card";
    const thumb=document.createElement("canvas");thumb.width=72;thumb.height=72;thumb.setAttribute("aria-label","Recalled visual impression");drawImpression(thumb,memory.sketch);
    const label=document.createElement("span");label.textContent=`${Math.round(memory.similarity*100)}% similar`;
    const age=document.createElement("small");age.textContent=`${Math.floor(memory.age/4)}s since seen`;
    card.append(thumb,label,age);$("memoryCards").append(card);
  }
  if(!vm?.recalled?.length){const empty=document.createElement("p");empty.className="memory-empty";empty.textContent=vm?.count?"No close visual match yet.":"Impressions form as time passes.";$("memoryCards").append(empty);}
}
function setSenseView(value){senseView=value;$("perspectiveButton").textContent=value?"Return to overview":"Through their eyes";$("perspectiveButton").setAttribute("aria-pressed",String(value));$("mapHint").textContent=value?"Only visible surroundings · select a portrait to change eyes":"WASD move · drag to look around";mini.hidden=value;updateCameraLabel();}
function toggleBuild(value=!building){building=value;$("buildPanel").hidden=!value;$("buildButton").setAttribute("aria-expanded",String(value));$("stage").classList.toggle("building",value);hoverTile=null;if(value){selectResident("player",true);toast("Choose a piece, then click a tile beside you.");}updateBuildPalette();}
function updateBuildPalette(){
  if(!state?.recipes)return;
  if(!paletteBuilt){
    const icons={wall:"▥",stone_wall:"▦",floor:"▱",door:"▯",shelter:"⌂",remove:"×"};
    for(const [kind,recipe] of [...Object.entries(state.recipes),["remove",{label:"Remove",cost:{}}]]){
      const button=document.createElement("button");button.className="build-piece";button.dataset.piece=kind;
      const icon=document.createElement("span");icon.className="piece-icon";icon.textContent=icons[kind];
      const label=document.createElement("span");label.textContent=recipe.label;
      const cost=document.createElement("small");cost.textContent=kind==="remove"?"Recover materials":Object.entries(recipe.cost).map(([item,n])=>`${n} ${item}`).join(" + ");
      button.append(icon,label,cost);button.addEventListener("click",()=>{buildKind=kind;updateBuildPalette();});$("buildPalette").append(button);
    }paletteBuilt=true;
  }
  const inventory=state.residents.find(a=>a.id==="player").inventory;
  document.querySelectorAll("[data-piece]").forEach(b=>{b.classList.toggle("active",b.dataset.piece===buildKind);b.setAttribute("aria-pressed",String(b.dataset.piece===buildKind));b.classList.toggle("short",Object.entries(state.recipes[b.dataset.piece]?.cost||{}).some(([item,n])=>inventory[item]<n));});
  $("buildHint").textContent=hoverTile?placementCheck(hoverTile.x,hoverTile.y):`${buildKind==="remove"?"Remove a structure":`Place ${state.recipes[buildKind].label.toLowerCase()}`} on your tile or one tile beside you.`;
}
async function place(x,y){
  if(!state||state.publicDemo)return;
  const result=await post("/api/action",{verb:buildKind==="remove"?"dismantle":"build",...(buildKind==="remove"?{}:{kind:buildKind}),x,y});
  if(result&&state.paused)toast("Placement queued. Press Step or resume.");
}
function buildRoster(){
  const roster=$("roster");roster.replaceChildren();
  for(const a of state.residents.filter(r=>r.id!=="player")){
    const button=document.createElement("button");button.className="resident-tile";button.dataset.id=a.id;button.setAttribute("aria-label",`Inspect ${a.name}`);
    const portrait=document.createElement("canvas");portrait.width=48;portrait.height=48;portrait.setAttribute("aria-hidden","true");drawCreature(portrait.getContext("2d"),24,39,{...a,action:"Resting"},0,1.25,"south");
    const name=document.createElement("span");name.textContent=a.name;
    const model=document.createElement("small");model.textContent=state.learning?.[a.id]?.backend==="laya-npu"?"Laya · NPU":"PPO";
    button.append(portrait,name,model);button.addEventListener("click",()=>selectResident(a.id,true));roster.append(button);
  }rosterBuilt=true;
}
const needNames={food:"Fullness",water:"Hydration",energy:"Energy",warmth:"Warmth",health:"Health",pain:"Pain",hunger_discomfort:"Hunger distress"};
for(const [key,label]of Object.entries(needNames)){
  const div=document.createElement("div");div.className="need";div.innerHTML=`<div class="need-label"><span>${label}</span><span id="${key}Value">100</span></div><div class="need-track" role="meter" aria-label="${label}" aria-valuemin="0" aria-valuemax="100"><div id="${key}Fill" class="need-fill ${key}"></div></div>`;$("needs").append(div);
}
for(const item of ["food","wood","stone","seed","amber_fruit"]){const div=document.createElement("div");div.className="inventory-item";div.innerHTML=`<b id="inv-${item}">0</b><span class="material">${item==="amber_fruit"?"Amber":item==="seed"?"Seeds":item[0].toUpperCase()+item.slice(1)}</span>`;$("inventory").append(div);}
function updateInspector(){
  if(!state)return;const a=state.residents.find(r=>r.id===selected);if(!a)return;
  $("selectedName").textContent=a.name;$("selectedKind").textContent=a.id==="player"?"YOUR CHARACTER":state.learning?.[a.id]?.backend==="laya-npu"?"LAYA · LOCAL NPU MODEL":"RECURRENT PPO LEARNER";$("selectedAction").textContent=a.action;$("residentMark").textContent=a.id==="player"?"YOU":a.id==="laya"?"LA":String(Number(a.id.slice(1))+1).padStart(2,"0");
  const p=$("portrait").getContext("2d");p.clearRect(0,0,96,96);drawCreature(p,48,79,{...a,action:"Resting"},0,2.3,"south");
  for(const key of Object.keys(needNames)){const value=Math.round(a[key]??(["pain","hunger_discomfort"].includes(key)?0:100));$(key+"Value").textContent=value;$(key+"Fill").style.width=`${value}%`;$(key+"Fill").classList.toggle("low",["pain","hunger_discomfort"].includes(key)?value>25:value<25);$(key+"Fill").parentElement.setAttribute("aria-valuenow",value);}
  $("injuryStats").textContent=`${a.amber_eaten||0} amber eaten · ${a.thorn_contacts||0} thorn contacts · ${Math.round(a.damage_taken||0)} damage taken`;
  for(const [item,value]of Object.entries(a.inventory))$("inv-"+item).textContent=value;
  $("inventoryCount").textContent=`${Object.values(a.inventory).reduce((s,n)=>s+n,0)+(a.carried_cache?4+a.carried_cache.food:0)} / 12`;$("gatheredCount").textContent=a.gathered;$("sharedCount").textContent=a.shared;$("stepsCount").textContent=a.steps;
  document.querySelectorAll(".resident-tile").forEach(b=>{b.classList.toggle("active",b.dataset.id===selected);b.setAttribute("aria-pressed",String(b.dataset.id===selected));});$("selectPlayer").classList.toggle("active",selected==="player");
  updateSenses();
}
function updateUI(){
  $("residentCount").textContent=String(state.residents.filter(r=>r.id!=="player").length).padStart(2,"0");
  $("day").textContent=`Day ${state.day}`;$("weather").textContent=state.weather==="Rain"?"☂ Rain":"☀ Clear";
  const minutes=Math.floor((6+state.dayProgress*24)*60)%1440;$("clock").textContent=`${String(Math.floor(minutes/60)).padStart(2,"0")}:${String(minutes%60).padStart(2,"0")}`;
  $("pauseButton").textContent=state.paused?"▶ Resume":"Ⅱ Pause";$("pauseButton").setAttribute("aria-label",state.paused?"Resume world":"Pause world");$("pauseOverlay").hidden=!state.paused;$("stepButton").disabled=!state.paused;
  $("worldMode").textContent=state.publicDemo?"SHARED PREVIEW":state.paused?"PAUSED":state.speed===1?"EXPLORING":"FAST-FORWARD";
  document.querySelectorAll("[data-speed]").forEach(b=>{const active=String(state.speed)===b.dataset.speed;b.classList.toggle("active",active);b.setAttribute("aria-pressed",String(active));});
  const achieved=Number(state.achievedSpeed||0);$("speedReadout").textContent=state.paused?"Time is still":state.speed===1?"Normal time":`${achieved.toFixed(1)}× achieved`;
  if(state.economy)$("economyStats").textContent=`Shared valley: ${state.economy.foodAvailable} harvestable food · ${state.economy.growingCrops} growing crops`;
  $("tickReadout").textContent=`${state.tick.toLocaleString()} ticks`;
  const me=state.residents.find(a=>a.id==="player");$("packCount").textContent=`Pack ${Object.values(me.inventory).reduce((s,n)=>s+n,0)+(me.carried_cache?4+me.carried_cache.food:0)} / 12`;
  $("loadButton").disabled=!state.hasSave||state.publicDemo;
  if(state.savedAt)$("saveStatus").textContent="Saved locally · mortality off";
  $("controllerNotice").textContent=state.controllerMode==="independent-learning"?`${state.controller} · skills still developing`:state.controller;
  if(state.controllerError)$("controllerNotice").textContent=`World paused: ${state.controllerError}`;
  if(state.publicDemo){document.querySelectorAll("[data-action],[data-speed],#pauseButton,#stepButton,#saveButton,#buildButton,#placeAhead,#placeHere").forEach(b=>b.disabled=true);$("controllerNotice").textContent=`Shared preview · ${state.controller}`;$("mapHint").textContent="Drag to explore · select a resident to inspect";}
  updateBuildPalette();
  if(!rosterBuilt)buildRoster();updateInspector();updateCameraLabel();
  const key=state.events.map(e=>e.id).join(",");if(key!==journalKey){journalKey=key;$("journal").replaceChildren();for(const event of [...state.events].reverse().slice(0,7)){const li=document.createElement("li");li.textContent=event.text;const time=document.createElement("time");time.textContent=`Day ${1+Math.floor(event.tick/2400)} · tick ${event.tick.toLocaleString()}`;li.append(time);$("journal").append(li);}}
}
function toast(message){if(!message)return;$("toast").textContent=message;$("toast").classList.add("show");clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("toast").classList.remove("show"),3200);}
async function poll(){
  if(pollBusy)return;pollBusy=true;
  try{
    const response=await fetch(`/api/state?perspective=${encodeURIComponent(selected)}${terrainNeeded?"&terrain=1":""}`,{cache:"no-store"});if(!response.ok)throw new Error(`Server returned ${response.status}`);const next=await response.json();const now=performance.now();
    if(next.terrain){terrain=next.terrain;terrainNeeded=false;}
    for(const a of next.residents){const old=state?.residents.find(b=>b.id===a.id);const p=old?interpolated(old,now):a;const far=Math.abs(a.x-p.x)+Math.abs(a.y-p.y)>4;tracks.set(a.id,{fromX:far?a.x:p.x,fromY:far?a.y:p.y,time:now,duration:Math.max(40,now-lastPollTime)});}
    state=next;lastPollTime=now;
    const view=sensory();visibleTiles=new Set(view?view.observation.tiles.filter(t=>t.terrain>=0).map(t=>`${view.origin[0]+t.dx},${view.origin[1]+t.dy}`):[]);
    if(terrain&&lastSeed!==state.seed){buildTerrain();lastSeed=state.seed;}
    lastRevision=state.revision;
    if(lastMessageSerial!==null&&state.messageSerial!==lastMessageSerial){if(!/^Walking|Action queued|Looking around$/i.test(state.lastMessage||""))toast(state.lastMessage);else{$("toast").classList.remove("show");}}
    lastMessageSerial=state.messageSerial;$("connection").hidden=true;updateUI();
  }catch(error){$("connection").hidden=false;$("connection").classList.add("error");$("connection").textContent="The valley is offline. Start the local server, then this view will reconnect.";}
  finally{pollBusy=false;}
}
async function post(endpoint,payload){
  try{const response=await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok||result.ok===false){if(!(payload.verb==="move"&&result.message?.startsWith("One action is already queued")))toast(result.message||result.error||"That action couldn't be completed.");return false;}return result;}catch{toast("Couldn't reach the valley. Check the local server.");return false;}
}
async function control(command,value){const result=await post("/api/control",{command,value});if(!result)return;if(command==="load"){terrainNeeded=true;lastSeed=null;tracks.clear();camera.follow=true;}if(["save","load"].includes(command))toast(result.message||"World updated.");await poll();}
async function act(verb,direction){if(!state||state.publicDemo)return;if(selected!=="player")selectResident("player",true);const command={verb};if(verb==="tone")command.tone=$("toneSelect").value;if(direction)command.direction=direction;if(["give","drop","eat"].includes(verb))command.item=$("itemSelect").value;const result=await post("/api/action",command);if(result&&state.paused)toast("Action queued. Press Step or resume the world.");}
$("pauseButton").addEventListener("click",()=>state&&control(state.paused?"resume":"pause"));$("stepButton").addEventListener("click",()=>control("step"));
document.querySelectorAll("[data-speed]").forEach(b=>b.addEventListener("click",()=>control("speed",b.dataset.speed==="max"?"max":Number(b.dataset.speed))));
document.querySelectorAll("[data-action]").forEach(b=>b.addEventListener("click",()=>act(b.dataset.action,b.dataset.direction)));
$("saveButton").addEventListener("click",()=>control("save"));$("loadButton").addEventListener("click",()=>control("load"));
$("buildButton").addEventListener("click",()=>toggleBuild());$("closeBuild").addEventListener("click",()=>toggleBuild(false));
$("perspectiveButton").addEventListener("click",()=>setSenseView(!senseView));
$("placeAhead").addEventListener("click",()=>{const me=state.residents.find(a=>a.id==="player"),[dx,dy]=directions[me.facing];place(me.x+dx,me.y+dy);});
$("placeHere").addEventListener("click",()=>{const me=state.residents.find(a=>a.id==="player");place(me.x,me.y);});
$("followButton").addEventListener("click",()=>selectResident(selected,true));$("selectPlayer").addEventListener("click",()=>selectResident("player",true));
$("overviewButton").addEventListener("click",showEveryone);
$("focusMapButton").addEventListener("click",()=>{const expanded=document.body.classList.toggle("map-focused");$("focusMapButton").setAttribute("aria-pressed",String(expanded));$("focusMapButton").setAttribute("aria-label",expanded?"Restore resident panel":"Expand map");});
function zoom(delta){camera.zoom=Math.min(2.3,Math.max(.15,camera.zoom+delta));}$("zoomIn").addEventListener("click",()=>zoom(.15));$("zoomOut").addEventListener("click",()=>zoom(-.15));
canvas.addEventListener("wheel",e=>{e.preventDefault();zoom(e.deltaY<0?.1:-.1);},{passive:false});
canvas.addEventListener("pointerdown",e=>{canvas.setPointerCapture(e.pointerId);pointer={x:e.clientX,y:e.clientY};drag={x:e.clientX,y:e.clientY,cameraX:camera.x,cameraY:camera.y,moved:false};});
function pointerTile(e){const bounds=canvas.getBoundingClientRect();return{x:Math.floor(((e.clientX-bounds.left-viewWidth/2)/camera.zoom+camera.x)/TILE),y:Math.floor(((e.clientY-bounds.top-viewHeight/2)/camera.zoom+camera.y)/TILE)};}
canvas.addEventListener("pointermove",e=>{if(building){hoverTile=pointerTile(e);updateBuildPalette();}if(!drag||senseView)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>5){drag.moved=true;camera.follow=false;camera.x=drag.cameraX-dx/camera.zoom;camera.y=drag.cameraY-dy/camera.zoom;$("cameraCaption").textContent="OBSERVING THE VALLEY";}});
canvas.addEventListener("pointerleave",()=>{hoverTile=null;});
canvas.addEventListener("pointerup",e=>{if(drag&&!drag.moved&&state){const tile=pointerTile(e);if(building)place(tile.x,tile.y);else{const a=state.residents.find(a=>a.x===tile.x&&a.y===tile.y&&canSee(a.x,a.y));if(a)selectResident(a.id);else if(state.structures[`${tile.x},${tile.y}`]?.kind==="door"&&canSee(tile.x,tile.y))post("/api/action",{verb:"toggle_door",x:tile.x,y:tile.y});}}drag=null;pointer=null;});
canvas.addEventListener("pointercancel",()=>{drag=null;pointer=null;});
mini.addEventListener("click",e=>{if(!state)return;const b=mini.getBoundingClientRect();camera.follow=false;camera.x=(e.clientX-b.left)/b.width*state.size*TILE;camera.y=(e.clientY-b.top)/b.height*state.size*TILE;$("cameraCaption").textContent="OBSERVING THE VALLEY";});
$("helpButton").addEventListener("click",()=>{$("helpDialog").showModal();pressed.clear();});$("closeHelp").addEventListener("click",()=>$("helpDialog").close());
const movement={w:"north",ArrowUp:"north",d:"east",ArrowRight:"east",s:"south",ArrowDown:"south",a:"west",ArrowLeft:"west"};
window.addEventListener("keydown",e=>{if($("helpDialog").open||["SELECT","INPUT","TEXTAREA"].includes(e.target.tagName)||(e.target.tagName==="BUTTON"&&[" ","Enter"].includes(e.key)))return;const key=e.key.length===1?e.key.toLowerCase():e.key;if(movement[key]){e.preventDefault();if(!pressed.has(key)){pressed.add(key);camera.follow=true;act("move",movement[key]);}}else if(!e.repeat){if(key===" "){e.preventDefault();state&&control(state.paused?"resume":"pause");}if(key==="b"){e.preventDefault();toggleBuild();}if(key==="Escape")toggleBuild(false);const action={e:"gather",f:"eat",r:"drink",g:"give",q:"drop",o:"toggle_door",h:"revive",t:"make_seeds",p:"plant",j:"tap",k:"tone"}[key];if(action){e.preventDefault();act(action);}if(key==="?")$("helpDialog").showModal();}});
window.addEventListener("keyup",e=>pressed.delete(e.key.length===1?e.key.toLowerCase():e.key));window.addEventListener("blur",()=>pressed.clear());
setInterval(()=>{if(pressed.size&&state&&!state.paused){const key=[...pressed].at(-1);act("move",movement[key]);}},280);
poll();setInterval(poll,150);
