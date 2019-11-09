# fuzzypeergroups
Fuzzy Peer Group Algorithm 

Pentru fiecare pixel se aplica algoritmul pe o fereastra (dimensiune nxn). Pentru fiecare pixel vecin din fereastra respectiva:
	- Se calculeaza o functie de similaritate (fuzzy similarity function) care ia valori in [0,1] si denota gradul de similaritate intre pixelul vecin curent si pixelul central din fereastra
	- Se ordoneaza pixelii descrescator in functie de valoarea functiei de similaritate
	- Se considera o functie C (functie ce arata gradul de apartenenta al pixelului la "grupul de similaritate - peer group") care reprezinta chiar valorile functiei de similaritate ordonate. C este o functie descrescatoare
	- Se considera si o functie A (functie ce arata gradul de "asimilare de similaritate" pana la un anumit pixel) care se determina prin sumarea valorilor functiei de similaritate pana la un pixel curent.
		- Nu se foloseste direct A, ci o funcite L (L=L(A)) (pentru a cuantiza certitudinea frazei "A(F(i)) este o multime larga" sau "gradul de similaritate acumulata este suficient")
	- Pe baza celor functiilor introduse se estimeaza numarul optim de pixeli din "grupul de similaritate" (m_optim)
		- Pentru fiecare m in {1,...,n^2-1} se calculeaza certitudinea C_FR1(m) (Fuzzy Rule 1). 
			C_FR1(m) = C(F(m))*L(F(m))
		- Valoarea lui m optima va fi valoarea pentru care certitudinea (C_FR1) este maxima
	- Se calculeaza certitudinea C_FR2(m) (Fuzzy Rule 2) pentru pixelul central/principal (F0)
		C_FR2(F0) = C(F(m_optim))*L(F(m_optim))		
			(Obs: C_FR2(F0)=C_FR1(m_optim). Nu mai e nevoie de alt efort computational)
	- In functie de valoarea C_FR2(F0) se ia decizia daca pixelul central este sau nu rezultat din zgomot impulsiv. 
		daca C_FR2(F0) >= Ft atunci F0 nu este afectat de zgomot impulsiv
		altfel F0 este impuls si trebuie inlocuit prin VMF (filtrare vectoriala)
	- Procedura de netezire a zgomotului gaussian. Pixelul F0 este inlocuit cu suma ponderata doar a pixelilor vecini care apartin "grupului de similaritate" (F(i) pentru i={0,...,m_optim})
		Fout = (sum(FP(F(i))*F(i)), i=0..m_optim) / (sum(FP(F(i))), i=0..m_optim)
			FP(F(i)) = ponderea pixelului vecin (=p(F0, Fi)=F(i)=C(F(i))) ("gradul de apartenenta")
