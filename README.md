Proiect in cadrul cursului CAVA - UB, FMI, an universitar 2021-2022, student Stanciu Andrei Calin </br>
</br>
Instructiuni rulare:</br>
    * codul sursa se afla in evaluare/evalueaza_solutie.py</br>
    * se ruleaza fara nici un argument: "python3 evalueaza_solutie.py" , </br>
        direct din folderul evaluare (pentru a functiona path-urile relative)</br>
    * path-urile sunt configurate pentru linux, este posibil ca pentru windows/ orice alt sistem de operare, sa trebuiasca sa fie schimbate </br>
        (din evalueaza_solutie.py, variabilele globale DATA_PATH_CLASIC, DATA_PATH_JIGSAW, TEMPLATE_PATH</br>
                                                        PREDICTION_PATH_CLASIC, PREDICTION_PATH_JIGSAW)</br>
    * python 3.8.10, opencv-python (4.5.3.56), numpy (1.20.3), nici o alta dependinta</br>
    * codul face predictii pentru datele aflate in folderul test/clasic si test/jigsaw</br>
    * codul genereaza predictiile in formatul cerut in fisiere_solutie/Stanciu_Calin_331/clasic </br>
       respectiv in fisiere_solutie/Stanciu_Calin_331/jigsaw </br>
</br>
Alte informatii despre fisiere:</br>
    * in evaluare/templates/ se afla template urile (extrase semi-manual din datele de antrenare) pentru cifre/contur   </br>
       folosite de catre evalueaza_solutie.py</br>
    * in evaluare/aux_py/ se afla bucati de cod folosite in timpul rezolvarii temei, </br>
       de exemplu pentru gridsearch sau extragerea de template uri </br>
       - aceste bucati de cod au fost introduse in scop demonstrativ, fara a fi folosite </br>
        	in mod direct pentru generarea predictiilor de catre evalueaza_solutie.py</br>
