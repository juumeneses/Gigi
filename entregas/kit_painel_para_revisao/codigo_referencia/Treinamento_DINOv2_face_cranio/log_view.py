"""Isolated log viewer; log contents are always rendered as escaped text."""
import html
import json


def log_html(content, follow, identity):
    key = json.dumps('dino-log-'+identity).replace('<', '\\u003c')
    return '''<!doctype html><html><head><meta charset="utf-8">
<style>body{margin:0}pre{box-sizing:border-box;margin:0;height:240px;overflow:auto;
padding:12px;background:#f6f7f9;color:#24262b;border:1px solid #d5d8df;
border-radius:4px;font:13px/1.6 ui-monospace,monospace;white-space:pre-wrap;
overflow-wrap:anywhere;letter-spacing:0}</style></head><body>
<pre id="log" tabindex="0" aria-label="Log da execucao">'''+html.escape(content)+'''</pre>
<script>
const box=document.getElementById('log');
const key='''+key+''';
const follow='''+json.dumps(bool(follow))+''';
let saved=0;
try {saved=Number(sessionStorage.getItem(key)||0);} catch(e) {}
box.scrollTop=follow ? box.scrollHeight : saved;
box.addEventListener('scroll',()=>{
  if(!follow) {try {sessionStorage.setItem(key,String(box.scrollTop));} catch(e) {}}
});
</script></body></html>'''
