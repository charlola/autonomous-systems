# autonomous-systems

Setup uniform Python-Environment using conda:  
conda create --name py38 python=3.8

Activate the new env:  
(Windows): activate py38  
(Mac/Unix): source activate py38


## Unity Setup
1. Anaconda runterladen:
1. Virtuelle Umgebung mit Python 3.6 anlegen:
conda create -n ml-agents python=3.6
1. Virtuelle Umgebung starten
activate ml-agents
1. In ein Ordner für Repositories wechseln und ml-agents downloaden: (dauert länger)
git clone --branch release_2 https://github.com/Unity-Technologies/ml-agents.git
1. In das ml-agents Repository wechseln
cd ml-agents
1. Das ml-agents envs Package installieren (muss vor ml-agents installiert werden)
cd ml-agents-envs
pip install -e .
1. Das ml-agents Package installieren (dauert länger)
cd ../ml-agents
pip install -e .

1. Als Administrator in Unity Hub Bereich "Installs"  Unity Version "2018.4.35f1" installieren (dauert länger)
1. In Unity Hub Bereich "Projects" "Add" drücken und den Ordner "ml-agents/Project" öffnen
1. Projekt öffnen (dauert länger)

Assets/Ml-Agents/Examples/Worm/Scenes WormDynamicTarget.unity öffnen und auf Play drücken.

Done!

