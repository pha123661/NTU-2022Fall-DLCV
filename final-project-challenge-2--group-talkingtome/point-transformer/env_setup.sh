echo "[PT INFO] Dependecies..."
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install h5py pyyaml sharedarray tensorboardx plyfile
echo "[PT INFO] Done !"
echo "[PT INFO] Installing cuda operations..."
cd lib/pointops
    python3 setup.py install
cd ../..
echo "[PT INFO] Done !"

NVCC="$(nvcc --version)"
TORCH="$(python -c "import torch; print(torch.__version__)")"

echo "[PT INFO] Finished the installation!"
echo "[PT INFO] ========== Configurations =========="
echo "$NVCC"
echo "[PT INFO] PyTorch version: $TORCH"
echo "[PT INFO] ===================================="