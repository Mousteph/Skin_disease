# Projet MLBIO : Classification des maladies de la peau

Bienvenue au projet de classification des maladies de la peau ! Le but de ce projet est de classifier différentes maladies de la peau en utilisant le jeu de données HAM10000 et de fournir une explication de la décision du modèle en utilisant la bibliothèque LIME.

Pour atteindre cet objectif, nous avons utilisé l'apprentissage par transfert avec un modèle ResNet18 pré-entraîné. Nous avons affiné le modèle en utilisant PyTorch et obtenu de bons résultats sur la tâche de classification.

### Serveur

Ce projet comprend un serveur qui permet aux utilisateurs de classifier les maladies de peau en envoyant des images au serveur. Le serveur utilise le modèle entraîné pour classer les images et renvoie à l'utilisateur la classe prédite, sa probabilité et une explication de la décision du modèle. Le serveur peut être lancé à l'aide de la commande : 

```bash
docker-compose up -d.
```

### Interface graphique

En plus du serveur, ce projet comprend également une interface graphique construite avec Streamlit, qui peut être lancée avec la commande : 

```bash
streamlit run src/frontend.py
```

L'interface graphique (http://localhost:8501/) permet aux utilisateurs de classifier des images en les téléchargeant via un navigateur web et de visualiser l'explication de la décision du modèle.

![alt text](images/frontend.png "Interface graphique")

### Command-line interface

Les utilisateurs peuvent également classifier les images en utilisant la command-line interface (CLI) en exécutant *client.py*. La CLI s'utilise de la façon suivante : 

```bash
client.py [-h] [--explain] [--precision PRECISION] image [image ...]
```

Les arguments sont les suivants :
- --explain : Fournit une explication détaillée de la prédiction du modèle.
- --precision PRECISION : Définit la précision de la probabilité dans la sortie. Les valeurs valides sont *Faible*, *Moyenne* et *Importante*. Une précision plus élevée donnera des résultats plus précis, mais augmentera également le temps d'exécution.
- image : Le chemin vers l'image ou les images à classifier.

Par exemple, pour classer une image *test.jpg* avec une explication et une précision *Importante*, vous pouvez utiliser la commande suivante :

```bash
python client.py --explain --precision Importante test.jpg
```

### API

Les utilisateurs peuvent également classifier des images à l'aide de l'API en effectuant une requête POST vers l'endpoint ""http://127.0.0.1:8089/predict" avec l'image jointe. Pour faire une demande à l'API, vous devez envoyer un objet JSON avec les champs suivants :

| Champ | Description | Obligatoire |
| ----- | ----------- | ----------- |
| image | L'image à classifier, encodée en base64. | Oui |
| explain | Si vous souhaitez inclure une explication dans la réponse. (bool) | Non |
| precision | La précision souhaitée. Les valeurs valides sont *Faible*, *Moyenne* et *Importante*. | Non |

Voici un exemple de demande à l'API à l'aide de la commande curl :

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "image": "base64_encoded_image_data",
  "explain": true,
  "precision": "high"
}' http://127.0.0.1:8089/predict
```


En cas de réussite, le serveur renvoie un objet JSON au format suivant :

```json
{
  "success": true,
  "prediction": "Melanocytic nevi",
  "probability": 0.95,
  "explain": "explanation_image" // Array
}
```

## Crédits

Le modèle ResNet18 pré-entraîné a été obtenu à partir de la bibliothèque torchvision. Le jeu de données HAM10000 a été récupéré à partir de [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000).