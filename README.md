This repo is based on [SMPLX](https://github.com/vchoutas/smplx.git).
To get a smplx mesh without flame parameters:
```python
python3 infer.py ./data/30min_data_0/output-output-smpl-3d/mesh-smplx/000000.npz ./data/30min_data_0/output-output-smpl-3d/smplfull/000000.json
```
You will get a mesh called 'smplx.ply' in the workspace dir. 
You can open smplh mesh './data/30min_data_0/output-output-smpl-3d/mesh/000000.ply' to compare with the infer result.

To write smplh mesh:
```python
python3 scripts/write_smplh_mesh.py --data_root data/30min_data_0
```

To write cano smplh mesh:
```python
python3 scripts/write_smplh_mesh.py --data_root data/30min_data_0 --cano
```

To fit cano smplx:
```python
python3 -m transfer_model --exp-cfg config_files/smplh2smplx.yaml
```

To write smplh initialized smplx mesh:
```python
python3 scripts/write_pre_smplx_mesh.py --data_root data/30min_data_0 --vis
```

To fit smplx:
```python
python3 -m transfer_model --exp-cfg config_files/30min_data_0/smplh2smplx-0000-6000-1.yaml
```