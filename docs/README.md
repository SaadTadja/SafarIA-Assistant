# docs/

Placez ici `demo.gif` (10-15 s, < 5 Mo), puis décommentez la ligne d'image en haut
du README principal.

Conversion depuis un enregistrement OBS :

    ffmpeg -i demo.mp4 -vf "fps=12,scale=900:-1:flags=lanczos" -loop 0 demo.gif
