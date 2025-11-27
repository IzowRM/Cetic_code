@echo off
REM Se placer dans le dossier du script (ton repo Git)
cd /d "%~dp0"

REM Construire le message de commit avec la date et l'heure Windows
set "MSG=Commit du %date% %time:~0,8%"

REM Lancer Git Bash dans ce dossier et exécuter les commandes Git
"C:\Program Files\Git\git-bash.exe" --cd="%~dp0" -c "echo 'Message de commit : %MSG%'; git add .; read -p 'Appuyer sur Entrée pour faire le commit...'; git commit -m \"%MSG%\"; read -p 'Appuyer sur Entrée pour faire le push...'; git push; echo 'Terminé.'; read -p 'Appuyer sur Entrée pour fermer...'"
