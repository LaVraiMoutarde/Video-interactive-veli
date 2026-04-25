@echo off
echo Lancement de l'Installation Interactive...
python main.py
if %errorlevel% neq 0 (
    echo.
    echo Une erreur est survenue lors du lancement.
    echo Verifiez que Python est bien installe et que les dependances sont ajournees.
    pause
)
