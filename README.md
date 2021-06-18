# autonomous-systems

Setup uniform Python-Environment using conda:  
conda create --name ml-agents python=3.6

Activate the new env:


## Setup
1. Anaconda runterladen:
1. Virtuelle Umgebung mit Python 3.6 anlegen:

> conda create -n ml-agents python=3.6

    (Windows): conda activate ml-agents  
    (Mac/Unix): source activate ml-agents

1. Virtuelle Umgebung starten
activate ml-agents

1. In ein Ordner für Repositories wechseln und ml-agents downloaden: (dauert länger)

> git clone --branch release_2 https://github.com/Unity-Technologies/ml-agents.git

1. In das ml-agents Repository wechseln
> cd ml-agents

1. Das ml-agents envs Package installieren (muss vor ml-agents installiert werden)

>cd ml-agents-envs

>pip install -e .

1. Das ml-agents Package installieren (dauert länger)

> cd ../ml-agents

> pip install -e .

1. Die Verbindung zwischen Gym und Unity 

> cd ../gym-unity

> pip install -e .

1. Pytorch installieren (dauert länger)

> conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

1. Falls eine Grafikkarte der Nvidia RTX 3000 Reihe verwendet wird

> pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Hier noch das Unity setup falls die Executatble nicht mehr läuft 

1. Als Administrator in Unity Hub Bereich "Installs"  Unity Version "2018.4.35f1" installieren (dauert länger)

1. In Unity Hub Bereich "Projects" "Add" drücken und den Ordner "ml-agents/Project" öffnen

1. Projekt öffnen (dauert länger)

Assets/Ml-Agents/Examples/Worm/Scenes WormDynamicTarget.unity öffnen und auf Play drücken.

Done!

For recording with Tensorboard
> tensorboard dev upload --logdir = runs
