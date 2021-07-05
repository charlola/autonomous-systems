# autonomous-systems

Setup uniform Python-Environment using conda: (open Anaconda Prompt)  
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

# Einige Aufrufe

## Beispiel Aufruf zum Trainieren

Alle Parameter mit den Default-Werten und Erklärung findet man in commandline.py 

Die verschiedenen Algorithmen können Beispielweise mit dem 'algorithmus' Parameter definiert werden. 
Hier ist zu beachten, dass 'appo' und 'aa2c' die entscheidenen Algorithmen sind, da sie Netze verwenden, die nicht nur den Mittelwert lernen, sondern auch die entsprechende Distribution der Aktionen. 'a2c' und 'ppo' verwenden primitivere Netze, die jedoch für einfache Environments auch ausreichen. 
```--algorithm <apo|a2c|appo|aa2c>```

> python main.py --algorithm appo --critic_lr 3e-4 --actor_lr 1e-4 --gamma 0.995 --normalize reward  --clip 0.2 --batch_size 5000 --mini_batch_size 5000 --ppo_episodes 3 --gae_lambda 0 --hidden_units "128 128" --advantage advantage

## Beispiel Aufruf zum weiter Trainieren eines alten Durchlaufs

Ein alter Durchlauf kann durch den load Parameter wieder geladen werden 
```--load run_<run_id>/<version> ```

> python main.py --algorithm appo --critic_lr 3e-4 --actor_lr 1e-4 --gamma 0.995 --normalize reward  --clip 0.2 --batch_size 5000 --mini_batch_size 5000 --ppo_episodes 3 --gae_lambda 0 --hidden_units "128 128" --advantage advantage --load run_074/final 

## Beispielaufruf um den trainierten Wurm zu sehen

Ein alter Durchlauf muss durch den load Parameter geladen werden 
```--load run_<run_id>/<version>```

Außerdem muss der Modus auf test gestellt werden (bzw. irgendetwas gestellt werden, das nicht 'train' is)
```--mode test```

> python main.py --algorithm appo --critic_lr 3e-4 --actor_lr 1e-4 --gamma 0.995 --normalize reward  --clip 0.2 --batch_size 5000 --mini_batch_size 5000 --ppo_episodes 3 --gae_lambda 0 --hidden_units "128 128" --advantage advantage --load run_074/final --mode test


# Hier noch das Unity setup falls die Executatble nicht mehr läuft 

1. Als Administrator in Unity Hub Bereich "Installs"  Unity Version "2018.4.35f1" installieren (dauert länger)

1. In Unity Hub Bereich "Projects" "Add" drücken und den Ordner "ml-agents/Project" öffnen

1. Projekt öffnen (dauert länger)

1. Assets/Ml-Agents/Examples/Worm/Scenes WormDynamicTarget.unity öffnen und auf Play drücken.

1. Wenn das läuft, dann kann man die Environment mit File->Build Settings->build die Environment exportieren.

Done!

