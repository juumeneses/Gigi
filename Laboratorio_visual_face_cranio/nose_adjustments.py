"""AI visual proposals from contact sheets, in recorded display coordinates.

These are not specialist annotations. Preserve uncertainty and source lineage.
"""
from pathlib import Path
import argparse
import copy
import numpy as np
from .core import load, save, sha, new_run, regions_for

# Display coordinates in each 360x230 tile of the saved nose sheets.
# Only the visible outer alar limits are revised; no symmetry completion.
ALAR = [
    (124,105,237,103), (133,107,230,102), (128,108,233,103), (130,109,232,107),
    (127,110,235,108), (131,107,231,106), (126,114,236,109), (128,106,234,103),
    (128,111,235,108), (130,107,232,105),
    (127,108,235,103), (133,111,230,106), (130,110,234,104), (130,117,233,113),
    (126,109,235,104), (130,112,233,111), (126,113,237,111), (130,110,232,107),
    (130,114,234,108), (129,111,233,106),
    (128,106,233,102), (129,108,234,105), (128,111,231,109), (130,109,231,105),
    (128,111,233,108), (131,111,230,109), (132,115,231,111), (129,112,234,111),
    (128,111,233,108), (128,109,232,112),
    (126,106,236,105), (128,108,233,105), (129,108,233,106), (128,113,234,109),
    (130,111,233,109), (129,109,232,108), (130,113,233,107), (130,115,232,108),
    (128,109,234,108), (130,110,232,109),
    None, None, None, None, None,
    (130,108,232,105), (126,120,235,117), (129,110,232,108), None, None,
]

# Nasal base, image-left opening limit, image-right opening limit, respectively.
# Coordinates refer to the independent 360x300 skull context tiles.
SKULL_NASAL = [
    (179,187,156,160,202,160),(189,181,160,152,218,151),
    (174,187,144,165,203,162),(180,182,156,165,207,162),
    (181,158,157,144,209,145),(179,158,157,143,200,143),
    (181,186,157,164,206,167),(187,195,168,178,208,178),
    (181,187,156,163,207,165),(177,194,151,176,204,175),
    (179,200,154,182,205,181),(174,212,150,190,196,189),
    (182,148,151,123,211,123),(177,196,155,181,201,181),
    (182,199,157,182,208,182),(178,187,147,158,209,158),
    (175,163,146,145,202,143),(180,203,154,182,207,182),
    (183,174,149,149,215,149),(172,149,146,130,197,130),
    (185,145,151,122,218,122),(179,168,153,146,206,145),
    (167,163,142,143,195,140),(170,135,148,118,192,118),
    None,(171,147,147,132,195,132),
    (165,149,137,131,194,129),(178,173,155,156,201,156),
    (163,181,136,161,193,160),(158,174,135,154,182,154),
    (192,159,163,140,219,137),(170,175,145,154,195,153),
    (184,163,156,142,213,142),(180,155,158,140,204,140),
    (157,137,132,118,183,118),(177,184,148,162,206,162),
    None,(181,165,157,146,205,145),
    (184,175,160,158,209,157),(177,124,148,103,206,103),
    (182,149,160,131,204,131),(190,148,166,130,214,130),
    (177,149,151,128,205,127),(191,167,167,148,215,148),
    (202,163,176,145,227,145),None,
    (179,147,148,126,210,126),(178,161,149,140,207,140),
    (191,158,165,140,216,140),None,
]


def adjust(source):
    source = Path(source).resolve()
    mapping_path = source/'face_sheet_mapping.json'
    if not mapping_path.exists():
        mapping_path = source/'sheet_mapping.json'
    if sha(mapping_path) != '81751d6ce8e6eef2cf0820a0c2df196b6c30edbb9f34f93e549e0bfc00e1f85f':
        raise ValueError('Ajustes visuais pertencem somente ao lote e transformações inspecionados')
    if sha(source/'cranio_context_mapping.json') != '6b2abffdb5e5d1081ec2ebb27742de7259846fbfcdf6141294f67235b83dc1e3':
        raise ValueError('Contexto craniano diferente do inspecionado')
    mapping = {m['key']:m for m in load(mapping_path)}
    skull_mapping = {m['key']:m for m in load(source/'cranio_context_mapping.json')}
    run = new_run('annotation_batches')
    config = load(source/'config.json')
    config.update(parent_batch=str(source), refinement='face_skull_visual_nose_v2',
                  correction_script_sha256=sha(__file__), annotation_author='Codex AI',
                  protocol='outer_visible_alar_extremes_no_symmetry', anatomical_validation=False)
    save(run/'config.json',config)
    save(run/'inputs.json',load(source/'inputs.json'))
    changes=[]
    for path in sorted(source.glob('*/proposal.json')):
        row=copy.deepcopy(load(path))
        if sha(row['path']) != row['hash']:
            raise ValueError('Image changed')
        row.update(parent_proposal=str(path),parent_proposal_hash=sha(path))
        if row['modality']=='face':
            entry=mapping[row['key']]
            index=entry['index']
            coords=ALAR[index]
            matrix=np.vstack([entry['matrix'],[0,0,1]])
            evidence=source/f'face_nose_sheet_{index//10}.jpg'
            if not evidence.exists():
                evidence=source/f'nose_sheet_{index//10}.jpg'
            row['visual_audit']=dict(author='Codex AI', points_examined=[2,3,4],
                evidence=str(evidence),
                summary='Nariz conferido por IA; demais pontos do MediaPipe ainda não revisados. '+
                        ('Asas nasais ajustadas; sem validação especializada.' if coords else
                         'Resolução insuficiente para confirmar asas nasais; pontos 3/4 indeterminados.'))
            for point in row['points']:
                if point['numero'] in (3,4):
                    old=copy.deepcopy(point)
                    if coords:
                        offset=(point['numero']-3)*2
                        x,y,_=np.linalg.inv(matrix) @ [coords[offset],coords[offset+1],1]
                        if not (0<=x<row['width'] and 0<=y<row['height']):
                            raise ValueError('Correction outside image')
                        point.update(x=float(x),y=float(y),estado='PROPOSTA_AJUSTADA_IA',
                                     visibility='NAO_AVALIADO',method='visual_ia_outer_alar')
                    else:
                        point.update(x=None,y=None,estado='INDETERMINADO',visibility='INDETERMINADO',
                                     reason='Limite alar pouco definido na resolução disponível')
                    changes.append(dict(person=row['person'],numero=point['numero'],before=old,after=copy.deepcopy(point)))
                if point['numero']==2 and index in (0,30,34,39,45):
                    old=copy.deepcopy(point)
                    point.update(x=None,y=None,estado='INDETERMINADO',visibility='INDETERMINADO',
                                 reason='Pelos dificultam confirmar encontro columela/lábio na imagem')
                    changes.append(dict(person=row['person'],numero=2,before=old,after=copy.deepcopy(point)))
            row['regions']=regions_for(row['points'],row['width'],row['height'])
        else:
            entry=skull_mapping[row['key']]
            coords=SKULL_NASAL[entry['index']]
            matrix=np.vstack([entry['matrix'],[0,0,1]])
            row['visual_audit']=dict(author='Codex AI',points_examined=[2,3,4],
                evidence=str(source/f'cranio_context_{entry["index"]//10}.jpg'),
                summary='Região nasal craniana conferida por IA; demais pontos ainda não revisados. '+
                        ('Proposta reposicionada na abertura óssea.' if coords else 'Limites nasais indeterminados por artefatos/definição insuficiente.'))
            for point in row['points']:
                if point['numero'] not in (2,3,4):
                    continue
                old=copy.deepcopy(point)
                if coords:
                    offset=(point['numero']-2)*2
                    x,y,_=np.linalg.inv(matrix) @ [coords[offset],coords[offset+1],1]
                    if not (0<=x<row['width'] and 0<=y<row['height']):
                        raise ValueError('Skull correction outside image')
                    point.update(x=float(x),y=float(y),estado='PROPOSTA_AJUSTADA_IA',
                                 visibility='NAO_AVALIADO',method='visual_ia_piriform_aperture')
                else:
                    point.update(x=None,y=None,estado='INDETERMINADO',visibility='INDETERMINADO',
                                 reason='Artefatos/definição insuficiente para confirmar margens nasais')
                changes.append(dict(person=row['person'],modality='cranio',numero=point['numero'],before=old,after=copy.deepcopy(point)))
            row['refinement_status']='SKULL_NOSE_VISUAL_PROPOSAL'
            row['regions']=regions_for(row['points'],row['width'],row['height'])
        target=run/row['key']/'proposal.json'
        save(target,row)
        save(target.parent/'proposal.sha.json',dict(sha256=sha(target)))
    save(run/'changes.json',changes)
    save(run/'status.json',dict(state='CONCLUIDO',faces_examined=50,faces_alar_adjusted=43,
                               faces_alar_indeterminate=7,skulls_nasal_adjusted=46,
                               skulls_nasal_indeterminate=4,anatomical_validation=False))
    from .refinement_report import report
    report(run)
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source')
    print(adjust(parser.parse_args().source))
