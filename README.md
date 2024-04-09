This repo is based on [SMPLX](https://github.com/vchoutas/smplx.git).
To get a smplx mesh without flame parameters:
```python
python3 infer.py ./data/30min_data_0/output-output-smpl-3d/mesh-smplx/000000.npz ./data/30min_data_0/output-output-smpl-3d/smplfull/000000.json
```
You will get a mesh called 'smplx.ply' in the workspace dir. 
You can open smplh mesh './data/30min_data_0/output-output-smpl-3d/mesh/000000.ply' to compare with the infer result.