Proiect in cadrul cursului CAVA - UB, FMI, an universitar 2021-2022, student Stanciu Andrei Calin

Instructiuni rulare:
    * codul sursa se afla in evaluare/evalueaza_solutie.py
    * se ruleaza fara nici un argument: "python3 evalueaza_solutie.py" , 
        direct din folderul evaluare (pentru a functiona path-urile relative)
    * path-urile sunt configurate pentru linux, este posibil ca pentru windows/ orice alt sistem de operare, sa trebuiasca sa fie schimbate 
        (din evalueaza_solutie.py, variabilele globale DATA_PATH_CLASIC, DATA_PATH_JIGSAW, TEMPLATE_PATH
                                                        PREDICTION_PATH_CLASIC, PREDICTION_PATH_JIGSAW)
    * python 3.8.10, opencv-python (4.5.3.56), numpy (1.20.3), nici o alta dependinta
    * codul face predictii pentru datele aflate in folderul test/clasic si test/jigsaw
    * codul genereaza predictiile in formatul cerut in fisiere_solutie/Stanciu_Calin_331/clasic 
       respectiv in fisiere_solutie/Stanciu_Calin_331/jigsaw 

Alte informatii despre fisiere:
    * in evaluare/templates/ se afla template urile (extrase semi-manual din datele de antrenare) pentru cifre/contur   
       folosite de catre evalueaza_solutie.py
    * in evaluare/aux_py/ se afla bucati de cod folosite in timpul rezolvarii temei, 
       de exemplu pentru gridsearch sau extragerea de template uri 
       - aceste bucati de cod au fost introduse in scop demonstrativ, fara a fi folosite 
        	in mod direct pentru generarea predictiilor de catre evalueaza_solutie.py
