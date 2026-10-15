"""File quality inventory only. No landmark detection or identity comparison."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from .common import LAB_ROOT, REPO_ROOT, sha256_file, save_csv
from .region_pilot import image_input, json_save

ROOT = LAB_ROOT/'resultados/auditoria_banco'
VERSION = 'file_inventory_v1'


def inspect(path):
    image, status = image_input(path)
    if image is None:
        return dict(status='ERRO', motivo=status)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Comparable fixed inspection scale; these are indicators, not clinical thresholds.
    scaled = cv2.resize(gray, (max(1, round(w*min(1,512/max(w,h)))), max(1, round(h*min(1,512/max(w,h))))))
    flags = []
    if min(w,h) < 128:
        flags.append('RESOLUCAO_BAIXA')
    if float(gray.std()) < 10:
        flags.append('CONTRASTE_BAIXO')
    return dict(status='ATENCAO' if flags else 'OK_TECNICO', motivo=';'.join(flags),
                largura=w, altura=h, contraste=float(gray.std()),
                nitidez_512=float(cv2.Laplacian(scaled, cv2.CV_64F).var()),
                sha256=sha256_file(path), bytes=path.stat().st_size)


def run_audit(run, resume=False):
    run = Path(run)
    manifest = LAB_ROOT/'dataset/manifestos/manifesto_mestre.csv'
    config = dict(version=VERSION, manifest_sha256=sha256_file(manifest),
                  scope='Todas as entradas do manifesto mestre; nao inclui arquivos nao catalogados',
                  thresholds={'min_dimension':128, 'gray_std':10}, anatomy_validated=False)
    if run.exists() and not resume:
        raise ValueError('Execucao existente; use --resume ou outra pasta')
    run.mkdir(parents=True, exist_ok=True)
    if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != config:
        raise ValueError('Manifesto ou configuracao mudou; crie outra execucao')
    json_save(run/'config.json', config)
    frame = pd.read_csv(manifest, dtype=str).fillna('')
    rows = []
    for n, source in frame.iterrows():
        for view, column in [('face','imagem_face'),('CF','imagem_cf'),('CE','imagem_ce'),('CD','imagem_cd')]:
            relative = source[column]
            path = REPO_ROOT/relative
            relocated = False
            if relative and not path.is_file() and view != 'face':
                candidates = (REPO_ROOT/'Treinamento_cranios/cranios_normalizados_variantes').glob(f'*/{source["ano"]}/{view}/{path.name}')
                for candidate in candidates:
                    if sha256_file(candidate) == source.get(column+'_sha256'):
                        path = candidate
                        relocated = True
                        break
            row = dict(id=source['id_pessoa'], ano=source['ano'], vista=view,
                       caminho_manifesto=relative, caminho=str(path) if relative else '', realocado=relocated)
            cache = run/f'item_{n}_{view}.json'
            signature = sha256_file(path) if relative and path.is_file() else None
            old = json.loads(cache.read_text()) if resume and cache.exists() else {}
            if old.get('signature') == signature and old.get('metadata') == row and 'result' in old:
                result = old['result']
            else:
                try:
                    result = inspect(path) if relative else dict(status='ERRO', motivo='NAO_CATALOGADO')
                except Exception as error:
                    result = dict(status='ERRO', motivo=f'{type(error).__name__}: {error}')
                json_save(cache, dict(signature=signature, metadata=row, result=result))
            rows.append(dict(**row, **result))
        if n % 25 == 0:
            print(f'{n+1}/{len(frame)} registros; {len(rows)} arquivos verificados', flush=True)
    result = pd.DataFrame(rows)
    hashes = result.get('sha256', pd.Series('', index=result.index)).fillna('')
    result['conteudo_repetido'] = hashes.ne('') & hashes.duplicated(keep=False)
    save_csv(result, run/'inventario.csv')
    summary = dict(registros=len(frame), entradas=len(result), status=result.status.value_counts().to_dict(),
                   realocados=int(result.realocado.sum()), entradas_com_conteudo_repetido=int(result.conteudo_repetido.sum()),
                   vistas=result.groupby(['vista','status']).size().to_dict())
    summary['vistas'] = {f'{a}/{b}':v for (a,b),v in summary['vistas'].items()}
    json_save(run/'resumo.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def render_file_detail(row):
    import streamlit as st
    path = row.get('caminho', '')
    reason = row.get('motivo', '')
    st.write(f"Caso: {row['id']} · Vista: {row['vista']} · Status: {row['status']}")
    st.write('Motivo: ' + (reason or 'Nenhuma falha nas verificações básicas.'))
    if not path:
        st.warning('Sem imagem para exibir: o manifesto não informa um arquivo para esta vista. Isso não significa que uma imagem foi aberta e considerada defeituosa.')
        st.caption('Pendência de catálogo. A auditoria não confirma se o arquivo existe em outra pasta e não atribui outra imagem automaticamente.')
        return
    st.code(str(path), language=None)
    if not Path(path).is_file():
        st.error('Arquivo não encontrado no caminho registrado. Não é possível mostrar uma prévia.')
        return
    image, read_status = image_input(path)
    if image is None:
        st.error(f'Não foi possível abrir o arquivo: {read_status}.')
        return
    from .region_pilot_ui import display_frame
    st.image(display_frame(image), channels='BGR', width='stretch', caption=f"{row['id']} · {row['vista']} · original")
    st.write({name: row.get(name, '') for name in ('largura', 'altura', 'contraste', 'nitidez_512', 'conteudo_repetido')})


def render_report():
    import streamlit as st
    with st.expander('Auditoria do banco completo'):
        if st.button('Executar auditoria do banco', key='run_bank_audit',
                     help='Verifica arquivos do manifesto, leitura, resolução e duplicações. Preserva relatórios anteriores; não corrige vínculos nem treina modelos.'):
            import sys
            from .region_pilot_ui import execute
            destination = ROOT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            code = execute([sys.executable, '-u', LAB_ROOT/'scripts/auditar_banco.py',
                            '--run-dir', destination])
            if code == 0:
                st.success('Auditoria concluída. O novo relatório está disponível abaixo.')
            else:
                st.error('Auditoria interrompida. Consulte o log acima; os relatórios anteriores foram preservados.')
        runs = sorted(ROOT.glob('*/inventario.csv'), reverse=True)
        if not runs:
            st.info('Auditoria do banco ainda não concluída.')
            return
        chosen = st.selectbox('Relatório do banco', runs, format_func=lambda p:p.parent.name)
        frame = pd.read_csv(chosen).fillna('')
        st.caption('Escopo: manifesto mestre. Indicadores técnicos não confirmam anatomia; arquivos não catalogados não foram examinados. Nenhuma imagem foi excluída.')
        st.dataframe(frame.groupby(['vista','status']).size().rename('quantidade').reset_index(), hide_index=True)
        status = st.multiselect('Status dos arquivos', ['ERRO','ATENCAO','OK_TECNICO'], default=['ERRO','ATENCAO'])
        filtered = frame[frame.status.isin(status)]
        st.dataframe(filtered, hide_index=True)
        st.download_button('Baixar inventário completo', chosen.read_bytes(), 'inventario_banco.csv')
        if not filtered.empty:
            idx = st.selectbox('Exemplo do banco', filtered.index.tolist(), format_func=lambda i:f'{frame.loc[i,"id"]} · {frame.loc[i,"vista"]} · {frame.loc[i,"status"]}')
            render_file_detail(frame.loc[idx].to_dict())
        else:
            st.info('Nenhum registro corresponde aos status selecionados.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=ROOT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_audit(args.run_dir, args.resume)
