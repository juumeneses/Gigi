"""Image quality proxies for review, not automatic anatomical acceptance."""
from .portable import Path
import numpy as np
from PIL import Image

from .core import check_stop, csv_write, load, new_run, read_image, save, sha


def measure(path):
    im = read_image(path)
    width, height = im.size
    im.thumbnail((256,256), Image.Resampling.LANCZOS)
    gray = np.asarray(im.convert('L'), dtype=float)
    if min(gray.shape) < 3:
        raise ValueError('Imagem pequena demais para medir nitidez')
    lap = gray[:-2,1:-1] + gray[2:,1:-1] + gray[1:-1,:-2] + gray[1:-1,2:] - 4*gray[1:-1,1:-1]
    return dict(width=width,height=height,sharpness=float(lap.var()),
                dark_fraction=float((gray<=5).mean()),light_fraction=float((gray>=250).mean()))


def audit(run, stop=None, log=print):
    run = Path(run)
    people = [p for p in load(run/'cohort.json')['people'] if p['split'] in ('treino','validacao')]
    output = new_run('quality_audits')
    rows = []
    for person in people:
        for view, image in person['views'].items():
            if view == 'CE': continue
            check_stop(stop)
            row = dict(person=person['person'],year=person['year'],split=person['split'],view=view,path=image['path'])
            try:
                row['hash'] = sha(image['path'])
                row.update(measure(image['path']),state='OK')
                if sha(image['path']) != row['hash']: raise ValueError('Entrada alterada durante leitura')
            except (OSError,ValueError) as exc:
                row.update(state='ERRO',error=str(exc))
            rows.append(row)
        if len(rows)%100 < 5: log(f'Qualidade: {len(rows)} imagens verificadas', flush=True)
    thresholds = {}
    for view in sorted({r['view'] for r in rows}):
        train = [r['sharpness'] for r in rows if r['view']==view and r['split']=='treino' and r['state']=='OK']
        if train: thresholds[view] = float(np.quantile(train,.25))
    for row in rows:
        row['review_lower_sharpness'] = row.get('sharpness',float('inf')) < thresholds.get(row['view'],-1)
        row['review_small_image'] = row['state']=='OK' and min(row['width'],row['height']) < 224
    save(output/'config.json',dict(run=str(run.resolve()),cohort_hash=sha(run/'cohort.json'),code_hash=sha(__file__),
        threshold_policy='training_lower_quartile_by_view',thresholds=thresholds,
        warning='Proxies influenciadas pelo conteudo e fundo; nao descartam imagens. Teste final nao examinado.'))
    save(output/'records.json',rows); csv_write(output/'records.csv',rows)
    save(output/'status.json',dict(state='CONCLUIDO',n=len(rows),failures=sum(r['state']=='ERRO' for r in rows)))
    return output


def controls(launch,busy):
    import streamlit as st
    with st.expander('Qualidade das imagens: nitidez e resolucao'):
        runs = [p.parent for p in sorted((Path(__file__).parent/'artifacts/fusions').glob('*/cohort.json'),reverse=True)]
        st.caption('A nitidez pode afetar o matching. Esta triagem compara proxies por vista, usando o quartil inferior do treino como alerta; fundo e conteudo tambem afetam a medida. Nao exclui imagens nem acessa o teste final.')
        if runs:
            run = st.selectbox('Coorte para revisar qualidade',runs,format_func=lambda p:p.name)
            if st.button('Verificar qualidade do treino e validacao',disabled=busy()):
                launch('quality_audit',dict(run=str(run)))
        paths = sorted((Path(__file__).parent/'artifacts/quality_audits').glob('*/records.json'),reverse=True)
        if not paths: return
        path = st.selectbox('Auditoria de qualidade',paths,format_func=lambda p:p.parent.name)
        rows = load(path)
        only = st.checkbox('Somente alertas de qualidade',value=True)
        shown = [r for r in rows if not only or r['state']=='ERRO' or r['review_lower_sharpness'] or r['review_small_image']]
        st.write(f'{len(shown)} / {len(rows)} imagens')
        st.dataframe(shown,hide_index=True)
        if shown:
            i = st.selectbox('Imagem da triagem',range(len(shown)),format_func=lambda i:shown[i]['person']+' / '+shown[i]['view'])
            try: st.image(read_image(shown[i]['path']),width=420)
            except (OSError,ValueError) as exc: st.warning(str(exc))
        st.download_button('Baixar triagem CSV',(path.parent/'records.csv').read_bytes(),'qualidade.csv')
