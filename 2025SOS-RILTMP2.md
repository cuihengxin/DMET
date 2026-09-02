An Efficient scaled opposite-spin MP2 method for
periodic systems
Idan Haritan,† Xiao Wang,∗,‡ and Tamar Goldzak∗,†
†TheAlexanderKofkinFacultyofEngineering,Bar-IlanUniversity,RamatGan52900,Israel
‡DepartmentofChemistryandBiochemistry,UniversityofCaliforniaSantaCruz,SantaCruz,
CA95064,UnitedStates
E-mail: xwang431@ucsc.edu; tamar.goldzak@biu.ac.il
Abstract
We develop SOS-RILT-MP2, an efficient Gaussian-based periodic scaled opposite-spin
second-orderMøller-Plessetperturbationtheory(SOS-MP2)algorithmthatutilizestheresolution-
of-the-identity approximation (RI) combined with the Laplace transform technique (LT). In
our previous work [J. Chem. Phys. 157, 174112 (2022)], we showed that SOS-MP2 yields
better predictions of the lattice constant, bulk modulus, and cohesive energy of 12 simple
semiconductorsandinsulatorscomparedtoconventionalMP2andsomeoftheleadingdensity
functionals. In this work, we present an efficient SOS-MP2 algorithm that has a scaling of
O(N4) with the number of atoms N in the unit cell and a reduced scaling with the number of
k-points in the Brillouin zone. We implemented and tested our algorithm on both molecular
andsolid-statesystems,confirmingthepredictedscalingbehaviorbysystematicallyincreasing
thenumber ofatoms, the sizeof thebasis set, and thedensity ofk-point sampling. Using the
benzenemolecularcrystalasacasestudy,wedemonstratedthatSOS-RILT-MP2achievessig-
nificantlyimprovedefficiencycomparedtoconventionalMP2. Thisefficientalgorithmcanbe
usedinthefuturetostudycomplexmaterialswithlargeunitcellsaswellasdefectstructures.
1
5202
raM
62
]hp-mehc.scisyhp[
1v28402.3052:viXra

1 Introduction
Theelectronicstructureofextendedsystems,suchassolidsandotherperiodicstructures,isknown
to be a computationally complex task due to the effectively infinite number of atoms. Ab initio
quantumchemistrymethodsformoleculeshavealonghistoryofdevelopinghierarchicalandsys-
tematically improvable wavefunction-based methods. Starting from the mean-field Hartree-Fock
(HF) approximation, a range of methods are available, such as Møller-Plesset perturbation theory
(MP)andcoupledcluster(CC)methods.1,2 However,theirdevelopmentandapplicationinthema-
terials science and condensed matter physics communities have been limited, primarily due to the
computational challenges posed by extended systems. Instead, density functional theory (DFT)3,4
has been the cornerstone of electronic structure calculations for extended systems. DFT is ex-
act in principle and operates on electron density rather than many-body wavefunction, making it
a low-scaling theory compared to wavefunction-based methods. However, in practice, using the
commonly employed semilocal or hybrid functionals5 can lead to systematic errors in calculating
certain properties of materials, such as band gaps, van der Waals interactions, and spectroscopic
properties.6,7 In recent years, wavefunction methods, such as the periodic extensions of MP and
CC,havebeensuccessfullyappliedtocondensedphasematerials.8–21Nevertheless,theuseofthese
methods is still limited, and further development and applications to periodic systems, especially
forcomplexsystemswithlargeunitcells,suchsurfacereactions,vanderWaals2Dmaterials,and
point defects, are needed. Among these methods, second-order Møller-Plesset perturbation the-
ory (MP2) is the simplest wavefunction method that captures electron correlation effects, with a
computational scaling of O(N5), where N denotes the number of atoms in the system.22 As a way
to improve MP2 accuracy without additional cost, spin-component-scaled MP2 (SCS-MP2) was
introduced by Grimme.23 In SCS-MP2, the same spin (SS) and opposite spin (OS) contributions
to the MP2 correlation energy are separately scaled, leading to the expression for the SCS-MP2
correlationenergy:
Ecorr = c Ecorr +c Ecorr (1)
SCS−MP2 os os ss ss
2

For c = c = 1 one retains the MP2 correlation energy. Building on the SCS concept, various
os ss
SCS variants of ab initio methods have been developed and tested.24–34 Among them is the scaled
opposite-spin MP2 (SOS-MP2)24 that retains and scales only the opposite spin component of the
correlation energy, offering further simplifications and computational advantages relative to SCS-
MP2. TheSCS/SOSvariantsofMP2havebeenshowntooutperformconventionalMP2formany
molecularproperties.23,25–29 Forexample,SCS-MP2/SOS-MP2wasfoundtoprovideameanabso-
lute deviation (MAD) of 1.18/1.36 kcal/mol for heats of formation in the G2/97 set of molecules,
respectively, significantly better than MP2 (MAD of 1.77 kcal/mol) and the most popular DFT-
B3LYP (MAD of 2.12 kcal/mol).30 The scaling of OS and SS contributions can be motivated in
several ways, including a derivation based on modified perturbation theory.29,31 Based on empir-
ical optimization against CCSD(T) reaction energies, the optimal parameters of SS and OS for
thermochemicalmolecularpropertiesproposedfirstbyGrimmeare(c ,c ) = (1.2,0.33).23
os ss
In addition to its comparably good performance,24,29 SOS-MP2 can be performed with a re-
ducedO(N4)scalingusingtheresolutionoftheidentityapproximation(RI,alsoknownasdensity
fitting, DF) and the Laplace transform (LT) of the energy denominator35,36 (see Section 2 for de-
tails). For example, Distasio and Head-Gordon27 obtained accurate inter-molecular binding ener-
giesfortheS22dataset37 withtheO(N4)SOS(MI)-MP2methodwithreoptimizedOSparameters.
Although SCS/SOS methods have been extensively studied in molecular systems, their applica-
tion and development to real solids and nanostructures are still scarce. In our recent work, we
showed that the results of our periodic SCS-MP2 calculations were accurate for the thermochem-
istry properties of a set of 12 solids, with errors that are smaller than those of the leading density
functionals.38 Another recent work by Liang, Ye, and Berkelbach used SCS-MP2 to predict the
cohesiveenergiesofmolecularcrystals.39 ItshowedthatbyreoptimizingtheSCSparameters,one
canachieve7.5kJ/molaccuracyforthecohesiveenergiesinthe X23molecularcrystaldataset.40
This work presents a novel and efficient periodic SOS-MP2 algorithm based on the RI and
LT techniques, inspired by algorithms that were proposed for molecules,24 using atom-centered
Gaussian orbitals adapted for periodic systems. This algorithm, termed SOS-RILT-MP2, reduces
3

thescalingofconventionalMP2toO(N4)withthenumberofatomsperunitcellandtoO(N2)the
k
numberofk-pointssampledintheBrillouinzone. Previously,combiningRIandLTwasintroduced
in periodic (conventional) MP2 with atomic orbitals, but has not reduced the computational scal-
ing.41–43 On the other hand, quartic scaling MP2 algorithms that incorporate RI and LT have been
developedforperiodicsystems,butwerebasedonplane-wavebasissets.44,45 Usingatom-centered
offer
basis sets can various advantages over plane waves, such as allowing for straightforward all-
electron calculations, easy access to core states, and better convergence behaviors with respect to
thebasissetsize.46
The structure of this paper is as follows, in Section 2 we will introduce the SOS-RILT-MP2
formalism and algorithm. Section 4 presents the results of the accuracy and timing of the SOS-
RILT-MP2 algorithm in the molecular case for linear alkane chains with increasing length and for
two ionic semiconductors, and an example for the benzene molecular crystal. Lastly, we present
ourconclusionsinSection5.
| 2 Methods and algorithm |     |     |     |     |     |     |     |
| ----------------------- | --- | --- | --- | --- | --- | --- | --- |
We used the single-particle basis of crystalline Gaussian-based atomic orbitals (AOs). These are
linear combinations of atom-centered Gaussian orbitals adapted to the translational symmetry of
the crystal.12 With periodic boundary conditions and N crystal momenta k sampled from the
k
Brillouinzone,theMP2correlationenergycanbedecomposedintoitsspincomponentsasfollows:
(cid:88)′
1 (cid:88)
| Ecorr = − |     | Taka,bkb(ikak |     | |jk bk | ),  |     | (2a) |
| --------- | --- | ------------- | --- | ------ | --- | --- | ---- |
|           |     |               |     | i a j  | b   |     |      |
| os N3     |     | iki,jkj       |     |        |     |     |      |
k kikakjkb iajb
(cid:88)′
|           | 1   | (cid:88)(cid:104) |           | (cid:105)     |     |       |      |
| --------- | --- | ----------------- | --------- | ------------- | --- | ----- | ---- |
| c orr = − |     | a                 | k , bk −T | b k , ak ×(ik | |jk |       |      |
| E         |     | T                 | a b       | b a           | ak  | bk ), | (2b) |
| s s N3    |     | i                 | k i, j k  | i k i, j k    | i a | j b   |      |
|           |     |                   | j         | j             |     |       |      |
k kikakjkb iajb
where
)∗
|     |          |     | (ikak | |jk bk |     |     |     |
| --- | -------- | --- | ----- | ------ | --- | --- | --- |
|     | Taka,bkb | =   | i a   | j b    | .   |     |     |
(3)
|     | iki,jkj | ε   | +ε  | −ε −ε |     |     |     |
| --- | ------- | --- | --- | ----- | --- | --- | --- |
|     |         | aka | bkb | iki   | jkj |     |     |
4

Electron repulsion integrals (ERIs) are expressed in Mulliken notation (11|22). Throughout this
paper, we use i, j to refer to occupied orbitals and a,b virtual orbitals obtained from periodic
Hartree-Fock (HF), which we assume to be spin-restricted. The primed summation indicates con-
servation of crystal momentum, k + k − k − k = G, where G is a reciprocal lattice vector.
a b i j
In this work, Gaussian density fitting (GDF), in other words, resolution of the identity (IR) with
Gaussian-basedauxiliarybasissets,wasusedtoevaluateERIs.47TheSCS-MP2correlationenergy
will be given by Eq.1. For SOS-MP2 the same-spin scaling coefficient is c = 0, and the optimal
ss
opposite spin parameter was empirically found to be c = 1.3.24 In our recent work, we showed
os
thatthisoptimalparameterisalsovalidforcalculatingthepropertiesofthe12semiconductorsand
insulatorsthatwetested.38
Inthiswork,wedevelopanefficientalgorithmforperiodicSOS-MP2thatcombinestheRIand
LT techniques, which we name SOS-RILT-MP2. First, we will describe the LT algorithm for the
denominator of the orbital energy differences in Eq.3. We can replace the 1/x function using the
(cid:82)
LT with an infinite integral 1 = ∞ e−xtdt. In Laplace transformed MP2, the energy denominators
x 0
aretransformedas
(cid:90)
1 ∞
= e−(εaka +εbkb −εiki −εjkj )tdt (4)
ε +ε −ε −ε
aka bkb iki jkj 0
ThisLaplaceintegralisthendiscretizedintoaweightedsumofquadraturepoints. Therearemany
different algorithms to evaluate the optimal quadrature points for the numerical integral evalua-
tion.36,42,48 Almlo¨f and Haser were the first to propose an efficient scheme that involved directly
minimizing the sum of squares error of the quadrature in order to choose the optimal quadrature
points,36,49 and obtained a precision of micro-Hartree (µH) with a small number of quadrature
points. The integral boundaries for each molecule were chosen on the basis of its orbital energy
differences. Another algorithm proposed by Ayala, Kudin, and Scuseria introduced a logarithmic
transform of the Laplace integration variable.42 An advancement proposed by Takatsuka, Ten-no,
and Hackbusch is to minimize the Chebyshev norm of the quadrature error using the minimax
5

different
algorithm.48 We tried all of the above methods for a set of five molecules from the S22
set,37
as well as diamond as a representative solid-state system; see the SI for a numerical com-
parison of all the methods. We found that Takatsuka, Ten-no, and Hackbusch’s algorithm was the
mostefficientamongthethreemethods,havinganaccuracyofµHwithnomorethan6quadrature
points, and we chose it for our periodic development of Laplace transformed SOS-MP2 (together
withRI).
WewillelaboratehereontheapproximationproposedbyTakatsuka,Ten-no,andHackbusch.48
The Laplace transform of a reciprocal 1/x in a certain interval x ∈ [1,R] can be evaluated using
thefollowingnumericalquadrature:
|     |     | 1           | (cid:88)L |         |     |     |
| --- | --- | ----------- | --------- | ------- | --- | --- |
|     |     |             | =         | we−xtl, |     |     |
|     |     | ≈ E (x;w,t) |           |         |     | (5) |
|     |     | k           | l l       | l       |     |     |
x
l=1
with the roots t and the weights w. Multiplying the equation by 1/A will give the corresponding
|     | l   | l   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
approximation in an arbitrary interval y = Ax ∈ [A,AR], such that 1 ≈ E (y;w˜ ,t˜), where w˜ =
|     |     |     |     |     | k l l | l   |
| --- | --- | --- | --- | --- | ----- | --- |
y
w/A,t˜ = t/A. The minimax approximation is used to find the optimal parameters that minimize
| l l | l   |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
the Chebyshev norm for each particular R. The parameter R is the intrinsic range of the problem
andnotlimitedtomolecularsystems. Inthiswork,Laplacedecompositionisusedinconcertwith
RI,sowewilluseitforeachpairofoccupiedandvirtualenergydifference,suchthat:
|     | (cid:90) ∞ |     | (cid:88) √ | √   |     |     |
| --- | ---------- | --- | ---------- | --- | --- | --- |
e−(εaka −εiki )te−(εbkb −εjkj )tdt = w e−(εaka −εiki )tl w e−(εbkb −εjkj )tl (6)
|     |     |     |     | l   | l   |     |
| --- | --- | --- | --- | --- | --- | --- |
0
l
The interval will be for each pair of orbitals, such that ε − ε ∈ [E ,E ], and E =
|     |     |     |     | aka iki | min max | min |
| --- | --- | --- | --- | ------- | ------- | --- |
=
ε − ε ,E ε − ε , where ε ,ε are orbital energies of the lowest virtual
| LkL HkH | max maxkmax | minkmin | LkL HkH |     |     |     |
| ------- | ----------- | ------- | ------- | --- | --- | --- |
orbitalandthehighestoccupiedorbitalacrossallk-points,respectively,andε ,ε arethe
|     |     |     |     |     | maxkmax minkmin |     |
| --- | --- | --- | --- | --- | --------------- | --- |
maximum and minimum orbital energies across all k-points, respectively. The quadrature points
and their weights in the minimax approximation w,t are determined for R = E /E . The
|     |     |     | l   | l   | max | min |
| --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | =   | =   | =   |
Laplace-transformed MP2 energies are calculated using w˜ w/E ,t˜ t/E (i.e., A E ).
|     |     |     |     | l l min | l l min | min |
| --- | --- | --- | --- | ------- | ------- | --- |
The convergence of the quadrature is exponential regardless of the intrinsic range R , resulting
6

in a low number of grid points with a certain accuracy for all problems. The quadrature points
developed by these authors have been kindly made available to the community and a copy can be
foundonGitLab.50,51
As mentioned above, in this work we will incorporate RI together with LT for the opposite-
spinpartoftheMP2correlationenergy(orsimply,SOS-RILT-MP2)inEq.2a. Wegeneralizedthe
SOS-MP2 algorithm developed for molecules to periodic systems.24,48 The RI approximation is
GDF.47
used to calculate the ERIs using We will denote the auxiliary basis functions by P,Q so
thatwecanwritetheERIsas:
(cid:88)Naux
=
|     |     |     | (ikak | |jk | bk ) | BP  | BP     | ,      |     |     | (7) |
| --- | --- | --- | ----- | --- | ---- | --- | ------ | ------ | --- | --- | --- |
|     |     |     | i     | a   | j b  |     | ikiaka | jkjbkb |     |     |     |
P
where
(cid:88)naux
|     |     |     | BP     | =   |       | |Q)(P|Q)−1/2. |     |     |     |     |     |
| --- | --- | --- | ------ | --- | ----- | ------------- | --- | --- | --- | --- | --- |
|     |     |     |        |     | (ikak |               |     |     |     |     | (8) |
|     |     |     | ikiaka |     | i     | a             |     |     |     |     |     |
Q
Thus, the opposite-spin component of the MP2 correlation energy given in Eq.2a is expressed
utilizingtheRIandLTapproximationsas:
|     | (cid:88) (cid:88)′ | (cid:88) |     |     |     |     |     |     |     |     |     |
| --- | ------------------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
1
E co rr = − (ik ak |jk bk )∗(ik ak |jk bk )w e−(εaka −εiki )tle−(εbkb −εjkj )tl (9)
|     |     |     | i   | a j | b i | a   | j b | l   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
o s N3
k
|     | l kikakjkb   | iajb     |              |         |        |              |        |         |                  |       |     |
| --- | ------------ | -------- | ------------ | ------- | ------ | ------------ | ------ | ------- | ---------------- | ----- | --- |
|     | (cid:88)′    |          | (cid:88)Naux |         |        | (cid:88)Naux |        |         |                  |       |     |
|     | 1 (cid:88)   | (cid:88) |              |         |        |              |        |         |                  |       |     |
| =   |              |          |              | P BP    | )∗(    | Q            | BQ     | e−(εaka | −εiki )tle−(εbkb | −εjkj | )tl |
|     | −            |          | ( B          |         |        | B            |        | )w      |                  |       |     |
|     | N3           |          |              | i kiaka | jkjbkb | i kiaka      | jkjbkb | l       |                  |       |     |
|     | k l kikakjkb | iajb     | P            |         |        | Q            |        |         |                  |       |     |
|     |              |          |             |         |        |              |        |       |                  |       |    |
(cid:88) (cid:88)′ (cid:88) (cid:88)  (cid:88) 
|     | 1   |     |  |     |     |     |     |  |     |     |     |
| --- | --- | --- | ------ | --- | --- | --- | --- | ------ | --- | --- | --- |
= − w (B P )∗B Q e−(εaka −εiki )tl (BP )∗BQ e−(εbkb −εjkj )tl
|     |     |     | l   | kiaka |         |     |     |     | jkjbkb |     |     |
| --- | --- | --- | --- | ----- | ------- | --- | --- | --- | ------ | --- | --- |
|     | N 3 |     |     | i     | i kiaka |     |     |     | jkjbkb |     |     |
k
|     | l kikakjkb         | PQ       | ia  |     |     |     |     | jb  |     |     |     |
| --- | ------------------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     | (cid:88) (cid:88)′ | (cid:88) |     |     |     |     |     |     |     |     |     |
1
| =   | −            |     | w MPQ(t)MPQ(t) |     |      |     |     |     |     |     |     |
| --- | ------------ | --- | -------------- | --- | ---- | --- | --- | --- | --- | --- | --- |
|     |              |     | l              | l   | l    |     |     |     |     |     |     |
|     | N3           |     | kika           |     | kjkb |     |     |     |     |     |     |
|     | k l kikakjkb | PQ  |                |     |      |     |     |     |     |     |     |
where
(cid:88)
|     |     |     | M P Q (t | ) = | (B P | )∗B Q | e−(εaka | −εiki )tl. |     |     | (10) |
| --- | --- | --- | -------- | --- | ---- | ----- | ------- | ---------- | --- | --- | ---- |
l
|     |     |     | k ik a |     | i kiaka | i kiaka |     |     |     |     |     |
| --- | --- | --- | ------ | --- | ------- | ------- | --- | --- | --- | --- | --- |
ia
The implementation can be seen in the pseudo-code algorithm in Alg.1. To retrieve molecular
7

orbital (MO) energy pairs ε − ε , we start with a HF SCF calculation. Using this, we will
aka iki
determine the parameters E , E , and R needed to obtain the quadrature points for Laplace
min max
integration. ThenwewillcalculatetheRIERItensor BP inEq.8,andsubsequentlythe MPQ(t)
l
ikiaka kika
tensor in Eq.10. Lastly, we will loop over k-points using the conservation of crystal momentum,
| andcalculate Ecorr | (Eq. 9. |     |     |     |
| ------------------ | ------- | --- | --- | --- |
os
MPQ(t)
The highest scaling step in this SOS-RILT-MP2 algorithm is calculating the tensor
kika l
|     |     | N2N | N2  |     |
| --- | --- | --- | --- | --- |
in Eq.10. This step has a scaling of N N , where N represents the number of k-points
|     |     | k   | l aux o v | k   |
| --- | --- | --- | --------- | --- |
sampled in the Brillouin zone, N the number of quadrature points of the LT, N the number of
l aux
auxiliary basis functions, N the number of occupied MOs, and N the number of virtual MOs.
|     |     | o   |     | v   |
| --- | --- | --- | --- | --- |
In contrast, the highest scaling step in RI-MP2 is the calculation of the ERIs using RI shown in
Eq.7,andhasascalingofN3N2N2N
. ThiscomparisonshowsthattheuseoftheSOS-RILT-MP2
|     | k   | o v aux |     |     |
| --- | --- | ------- | --- | --- |
N5
algorithmreducestheformalscalingwiththenumberofatomsintheunitcellfrom inRI-MP2
to N4, as the number of quadrature points N to obtain a µH accuracy is small (see results below).
l
Italsohasareducedscalingwiththenumberofk-pointsfrom N3 to N2. Thisisagreatadvantage
k k
for modeling solids with complex unit cells, such as point defects which usually require a large
unitcellinordertoreducetheperiodicartifactinteractionsfromotherdefectsinneighboringcells.
In the following section, we present the results that compare SOS-RILT-MP2 with conventional
MP2(i.e.,RI-MP2).
Algorithm1PseudocodeforcalculatingSOS-RILTMP2
| CollectMOenergypairsε |     | −ε fromSCFcalculations |     |     |
| --------------------- | --- | ---------------------- | --- | --- |
aka iki
| Calculate E | ,E ,R |     |     |     |
| ----------- | ----- | --- | --- | --- |
| min         | max   |     |     |     |
UsingRforretrievingthegridpointsandweightst,w fortheLTquadrature
l l
| CalculatetheRI-ERItensor |     | BP  |     |     |
| ------------------------ | --- | --- | --- | --- |
ikiaka
Calculate MPQ(t)
l
kika
| loopoverk,k | ,k                                |     |     |     |
| ----------- | --------------------------------- | --- | --- | --- |
| i           | j a                               |     |     |     |
| Calculatek  | fromconservationofcrystalmomentum |     |     |     |
b
|       | = (cid:80) (cid:80) | MPQ(t)MPQ(t) |          |     |
| ----- | ------------------- | ------------ | -------- | --- |
| E ← E | − 1                 | w            |          |     |
| os os | N 3 l               | PQ l kika    | l kjkb l |     |
k
endloop
8

3 Computational Details
To evaluate the performance of the SOS-RILT-MP2 algorithm, we implemented it in PySCF.52,53
ToachievescalabilitybenefitsoverconventionalMP2,weappliedthealgorithmtobothmolecular
and solid systems. The molecular systems included a series of linear alkane chains of increasing
length: C H , C H , C H , C H , and C H . The solid-state systems included two insu-
10 22 20 42 30 62 40 82 50 102
lators: diamond (C) and aluminum nitride (AlN). These choices allowed us to rigorously assess
the algorithm’s performance in terms of accuracy and cost, examining the influence of increasing
number of atoms within the unit cells in the molecular alkane chain calculations, and the impact
ofanescalatingnumberofk-pointsandvirtualorbitalsinthesolidsystemcalculations. Addition-
ally, we tested a benzene molecular crystal to evaluate the algorithm’s performance on a complex
system with many atoms per unit cell and a large number of k-points, presenting a more challeng-
ing case for system size. Periodic boundary conditions were adopted for all the systems under
study. For the alkane molecules, we performed the calculations at the Γ point using a large unit
cell with a vacuum region of approximately 30 Å in all three dimensions, preventing interactions
betweenneighboringmoleculestoensureisolatedconditions. Forthesolidsystems,includingdia-
mond,AlN,andbenzenecrystal,weperformedthecalculationsusingk-pointgridswithauniform
Monkhorst-Packmeshrangingfrom N = 1×1×1to5×5×5intheBrillouinzonethatincludes
k
the Γ point. To mitigate the issue of the divergent exchange term in periodic Hartree-Fock (HF)
calculations, we used a Madelung constant correction scheme.54–56 This correction ensures that
boththetotalenergiesandtheorbitalenergiesconvergetowardthethermodynamiclimit(TDL)as
N−1,withexamplespresentedinourrecentpaper.38
k
Furthermore,weconductedtwodistinctsetsofcalculations: all-electroncalculationsandpseu-
dopotentialcalculations. Inthepseudopotentialcalculations,wetestedthemolecularalkanechains
and the three solid systems (diamond, AlN, and benzene crystal). In these calculations, we re-
placed the core electrons with Goedecker-Teter-Hutter (GTH) pseudopotentials57,58 and used the
GTH-cc-pVXZ correlation-consistent Gaussian basis sets that are optimized for periodic calcula-
tions with GTH pseudopotentials.59 Specifically, the double-zeta GTH-cc-pVDZ was used for the
9

molecularchainsgiventhelargenumberofatoms,uptotriple-zetaGTH-cc-pVXZ(X=D,T)were
used for benzene crystal to manage memory constraints, and up to quadruple-zeta GTH-cc-pVXZ
(X=D,T,Q)weretestedfordiamondandAlN.
In the all-electron calculations, we tested the diamond solid system and the molecular chains,
excluding C H due to computational constraints. In these calculations, we used Dunning’s
50 102
originalcc-pVXZbasisset.60 Again,thedouble-zetacc-pVDZwasusedforthemolecularchains,
anduptoquadruple-zetacc-pVXZ(X=D,T,Q)weretestedfordiamond. Forthecalculationofthe
ERIs in all basis sets, we used the Gaussian density fitting (GDF) method47 with the cc-pVDZ-RI
auxiliarybasisforthemoleculesandbenzenecrystalandcc-pVQZ-RIfordiamondandAlN.
All Calculations were preformed on Intel Xeon Gold 6338 205W processors @ 2.0 GHz with
4cores.
4 Results
Asafirststep,toverifytheaccuracyoftheLTpartintheSOS-RILT-MP2algorithm,wecompared
the opposite-spin energy term, Ecorr, in the SOS-RILT-MP2 algorithm (Eq. 2a) to that in the con-
os
ventionalMP2algorithm. SincebothSOS-RILT-MP2andconventionalMP2useRI,thedifference
betweenthemcanonlybeduetotheLTcomponent. Specifically,weexaminedtheabsoluteenergy
difference between these terms at different quadrature points customized to each calculation set.
For the set involving all-electron calculations, we expanded the range of quadrature points from 6
to19. Incontrast,forthesetusingGTHpseudopotentials,werestrictedouranalysistoquadrature
points between 6 and 13. Note that these evaluations for the solid systems were performed using
a 3x3x3 k-point grid. The findings, depicted in Fig. 1, reveal a consistent pattern in the tested
systems.
Across all panels, we observe that increasing the number of quadrature points exponentially
reduces the LT error (i.e., the energy difference between conventional MP2 and SOS-RILT-MP2)
in every system, as expected. In other words, with more quadrature points, the SOS-RILT-MP2
10

Molecules C AlN
(!, GTH-DZ) (3x3x3, GTH-[D/T/Q]Z) (3x3x3, GTH-[D/T/Q]Z)
10-4
10-6
10-6
10-6
)u
.a (|E 10-8 10-8 10-8
"
|
10-10
10-10
10-10
10-12
6 8 10 12 6 8 10 12 6 8 10 12
= = =
(a) (b) (c)
Molecules C
(!, AE-DZ) (3x3x3, AE-[D/T/Q]Z)
10-4
10-4 C H
10 22
C H
20 42
)u C 30 H 62
.a (|E10-6 10-6 C 40 H 82
" C H
| 50 102
dz
tz
10-8
10-8 qz
6 8 10 12 14 16 18 6 8 10 12 14 16 18
= =
(d) (e)
Figure 1: Log scaled absolute energy difference between the opposite spin correlation energies
(a.u)oftheSOS-RILTMP2algorithmandtheconventionalMP2asafunctionofdifferentquadra-
ture points. (a) Linear alkane chains with increasing lengths (C H to C H ) calculated using
10 22 50 102
the GTH pseudopotential and the GTH-cc-pVDZ basis set at the Gamma point. (b) Diamond cal-
culatedwiththeGTHpseudopotentialandtheGTH-cc-pVXZ(X=D,T,Q)basissetsusinga3x3x3
k-pointgrid. (c)AlNcalculatedwiththeGTHpseudopotentialandtheGTH-cc-pVXZ(X=D,T,Q)
basis sets using a 3x3x3 k-point grid. (d) Linear alkane chains with increasing lengths (C H
10 22
to C H ) calculated with the all-electron cc-pVDZ basis set at the Gamma point. (e) Diamond
40 82
calculatedwiththeall-electroncc-pVXZ(X=D,T,Q)basissetsusinga3x3x3k-pointgrid.
method steadily improves in precision. Practically, this improvement is so rapid that only a rela-
tivelysmallnumberofquadraturepointsisrequiredtoachieveaµHaccuracy. Thistrendholdsfor
alkane chains, diamond, and AlN. A similar convergence behavior was also observed for benzene
crystal, though not shown here. These findings highlight the robustness and consistency of the
SOS-RILT-MP2algorithmacrossawiderangeofchemicalstructures.
Examiningtheeffectofthebasissetsizeontheenergydifferenceinthesolids(panels(b),(c),
11

and(e)),weseethatalargerbasissetamplifiestheLTerrorinbothall-electronandpseudopotential
calculations. Thereasonisstraightforward: alargerbasisset,appliedtothesamesystem,includes
more orbitals and thus increases the maximum orbital energy across all k-points, ε . As a
maxkmax
result, E andRbothincrease,makingahigherquadraturepointnecessarytomaintainthesame
max
accuracy(seeEqs.5-6).
A similar size effect is evident for the alkene chains. In the all-electron calculations shown
in panel (d), increasing the system size, i.e., the chain length, further widens the LT error. This
occurs because a larger system increases ε −ε , and consequently E and R, leading
maxkmax mink min max
toahighernumberofquadraturepointsrequiredtopreservethesamelevelofaccuracy.
However, in the pseudopotential calculations in panel (a), there is a different behavior. There,
we can distinguish between two subsets: C H and C H versus C H , C H , and C H .
10 22 20 42 30 62 40 82 50 102
Within each subset, the trend holds: as system size increases, the LT error increases as well. Yet a
discrepancy appears when moving from C H to C H , where the difference decreases instead
20 42 30 62
of increasing. This sharp discontinuity arises from a numerical artifact in the calculations. Yet,
evenwiththisartifact,thesefindingsdemonstratethattheSOS-RILT-MP2methodachieveshighly
accurateresultsattheµH level,withasmallnumberofquadraturepoints.
Furthermore, looking at Fig. 1, we can notice an additional trend. Comparing panels (a) and
(b)withpanels(c)and(d)revealsthattheall-electroncalculationsdemandmorequadraturepoints
to reach the same level of LT error as in the pseudopotential calculations. For instance, in the
alkene chains, panel (a) shows that a quadrature point of 8 is suffices for an acceptable µH differ-
ence between the methods in the pseudopotential calculations, while panel (d) indicates a higher
quadraturepointof14isneededfortheall-electroncalculations. Thisdifferenceinoptimalquadra-
ture points can be attributed to the different orbital energies obtained in each type of calculation.
In the all-electron case, we calculate the core orbital energies, while in the pseudopotential case
wedonot. Therefore,intheall-electroncalculations,thecoreorbitalenergiesdrivedownε ,
mink
min
andhencepushup E andR. Asaresult,morequadraturepointsareneededtoachieveasimilar
max
levelofaccuracycomparedtothepseudopotentialcase.
12

Overall, these findings confirm the reliability and high accuracy at the µH level with a small
number of quadrature points for the SOS-RILT-MP2 method, while highlighting its potential for
broader application in complex computational chemistry scenarios. In particular, they highlight
themethod’sexcellentperformanceinbothall-electronandpseudopotentialcalculations,whichis
especiallyrelevantforproblemsinvolvingheavyelementsandcoreexcitations. Arecentstudyhas
shown that all-electron calculations significantly improve accuracy in properties such as the bulk
modulus of heavy-element solids, capturing crucial relativistic effects.61 Moreover, G W studies
0 0
based on Gaussian basis sets have further demonstrated the value of all-electron methods, accu-
rately determining quasiparticle energies, band structures, and both valence and core excitations
in weakly correlated materials.62 Taken together, these advances stress the importance of a robust
method, like SOS-RILT-MP2, that can handle both all-electron and pseudopotential calculations,
toenableamorecomprehensiveunderstandingofcomplexelectronicsystems.
Inthesecondpartofouranalysis,weaimedtoconfirmthescalingofthemosttime-consuming
component in each algorithm: that is, constructing the MPQ(t) tensor (referred to as the “M ten-
kika l
sor”, see Eq. 10 and Alg. 1) for SOS-RILT-MP2, and calculating the ERIs with the auxiliary basis
(referredtoasthe“oovvstep”,seeEq.7)forconventionalMP2.
We examined the time consumption for each step and the total time consumed by each algo-
rithm as a function of the system’s complexity: For the alkane chains, we focused on scaling as
the number of atoms increased. In solid systems, we explored the impact of the basis set size on
the computation time, specifically using a 3x3x3 k-point grid. It should be noted that, based on
Fig. 1, to achieve an acceptable µH difference in energy between the two algorithms for all of the
systems in question, we employed nineteen quadrature points for the all-electron calculations and
elevenquadraturepointsforthepseudopotentialcalculations. Notethatforeachsolid,thenumber
offunctionsintheauxiliarybasissetandthenumberofoccupiedorbitalsremainedconstantacross
the different basis sets, while only the number of virtual orbitals increased as the basis set grew
larger. In this case, because the oovv step scales as N3N2N2N (see Eq. 7), its time consumption
k o v aux
is expected to scale with an exponent of 2 relative to the basis set size, since only the number of
13

|       | Molecules   |                       | C   |     | AlN                   |     |     |
| ----- | ----------- | --------------------- | --- | --- | --------------------- | --- | --- |
|       | (!, GTH-DZ) | (3x3x3, GTH-[D/T/Q]Z) |     |     | (3x3x3, GTH-[D/T/Q]Z) |     |     |
| 3 . 3 | 7 2         | 0 . 7                 | 8 7 |     | 1 . 0 5 5             |     |     |
103
| 3 . 3 | 2 6 | 0 . 4 | 3 1 |     | 0 . 7 5 2 |     |     |
| ----- | --- | ----- | --- | --- | --------- | --- | --- |
| 4.493 |     | 1.931 |     |     | 2.050     |     |     |
102
| 4.298 |     | 1.811 |     |     | 2.187 |     |     |
| ----- | --- | ----- | --- | --- | ----- | --- | --- |
103
)s102
d
n
o c
e
s
( e
m
iT101
101
| 100   |              |                      |     | 102 |                      |         |     |
| ----- | ------------ | -------------------- | --- | --- | -------------------- | ------- | --- |
| 10    | 20 30 40     | 30                   | 50  | 100 | 60 100               | 150 200 |     |
|       | # of C atoms | # of basis functions |     |     | # of basis functions |         |     |
|       | (a)          |                      | (b) |     | (c)                  |         |     |
|       | Molecules    |                      | C   |     |                      |         |     |
|       | (!, AE-DZ)   | (3x3x3, AE-[D/T/Q]Z) |     |     |                      |         |     |
| 3.302 |              | 0.751                |     |     |                      |         |     |
| 3.286 |              | 0.380                |     |     |                      |         |     |
| 4.511 |              | 1.956                |     |     |                      |         |     |
| 4.300 |              | 2.000                |     |     |                      |         |     |
| 102   |              | 102                  |     |     |                      |         |     |
)s
d
| n   |     |     |     |     | M tensor  |     |     |
| --- | --- | --- | --- | --- | --------- | --- | --- |
| o   |     |     |     |     | SOS total |     |     |
c
| e s |     |     |     |     | oovv |     |     |
| --- | --- | --- | --- | --- | ---- | --- | --- |
( e
MP2 total
m
iT101
101
| 10  | 20 30 40     | 30                   | 50  | 100 |     |     |     |
| --- | ------------ | -------------------- | --- | --- | --- | --- | --- |
|     | # of C atoms | # of basis functions |     |     |     |     |     |
|     | (d)          |                      | (e) |     |     |     |     |
Figure 2: Log scaled time consumption (in seconds) for the M tensor step of the SOS-RILT MP2
algorithm (”M tensor”) , whole of the SOS-RILT MP2 algorithm (”SOS total”), the oovv step of
the conventional MP2 algorithm (”oovv”), and whole of the conventional MP2 algorithms (”MP2
total”), as function of different system complexities. (a) as a function of the number of carbon
atoms in linear alkane chains of increasing length (C H to C H ) calculated using a gth-
|     |     |     |     | 10 22 | 50 102 |     |     |
| --- | --- | --- | --- | ----- | ------ | --- | --- |
pseudopotential and an optimized cc-PVDZ basis set at the Gamma point (b) as a function of the
number of optimised basis functions (cc-pVDZ to cc-pVQZ) used calculating Diamond with a
gth-pseudopotential usinga 3x3x3 k-pointgrid (C) asa function ofthe number of optimisedbasis
functions (cc-pVDZ to cc-pVQZ) used calculating AlN with a gth-pseudopotential using a 3x3x3
k-point grid (d) as a function of the number of carbon atoms in linear alkane chains of increasing
length (C H to C H ) calculated using an all-electron approach and a cc-PVDZ basis set at
10 22 40 82
the Gamma point (e) as a function of the number of basis functions (cc-pVDZ to cc-pVQZ) used
calculating Diamond with an all-electron approach using a 3x3x3 k-point grid. For each data set,
trendlinesarepresented,withthelegendindicatingtheirslopes.
N2N N2
virtual orbitals (N ) changes. By contrast, for the M tensor, which scales as N N (see
| v   |     |     |     |     |     | k l aux | o v |
| --- | --- | --- | --- | --- | --- | ------- | --- |
Eq. 10), the time consumption is expected to scale with an exponent of 1 with respect to the basis
setsize. However,inthemolecularcase,thenumberoffunctionsintheauxiliarybasissetandthe
14

numberofoccupiedandvirtualmolecularorbitalsallincreaseasthemolecularchainlengthgrows.
Consequently, the time consumption for the oovv step is expected to scale with an exponent of 5
relative to the chain size, whereas the time consumption for the M tensor is anticipated to scale
withanexponentof4withrespecttothebasissetsize.
Theresultsforthispartofouranalysis(seeFig.2)confirmthetheoreticalpredictionsregarding
the scaling of the most time-intensive components in both algorithms, i.e. the M tensor and the
oovv step. In all of our systems (panels (a) to (e) in Fig.2), the M tensor scales at least one unit
less than the oovv step. A similar trend is also apparent for the total time consumed by each
algorithm,wherethetotaltimeconsumedbytheSOS-RILT-MP2algorithmscalesatleastoneunit
lessthanthetotaltimeconsumedbytheMP2algorithm. Oneofthebiggestdifferencesbetweenthe
algorithms can be seen in panel (e) in Fig.2, showing the time scaling for an all-electron diamond
calculation. In this panel, we can see that both the oovv step and the total time for conventional
MP2 scale significantly higher than the M tensor and the total time for SOS-RILT MP2, with a
slope of 1.956 and 2.000 vs. 0.751 and 0.380 respectively. These results demonstrate the superior
efficiencyandscalabilityoftheSOS-RILT-MP2algorithmincalculatingcomplexunitcellswitha
largesetofbasesinperiodicsystems.
To reinforce our conclusion on the efficiency of the SOS-RILT-MP2 algorithm, we tested it
on the benzene molecular crystal. Due to memory constraints, we used a 2×2×2 k-point grid
withGTH-cc-pVDZandGTH-cc-pVTZbasissets,anda3×3×3k-pointgridwithGTH-cc-pVDZ
basis set.59 The results in Table 1 clearly demonstrate SOS-RILT-MP2’s superior scalability. For
example,inthe3×3×3k-pointgridcase,theoovvstepinconventionalMP2tookabout276hours,
and the total time 364 hours. On the contrary, for the same case, the M tensor step in SOS-RILT-
MP2 took only 38 hours, and the total time 47 hours. These results highlight the significantly
improved efficiency of SOS-RILT-MP2, making it particularly effective for periodic systems, and
especiallythosewithlargeandcomplexunitcells.
In the last step of our analysis, which focused solely on the pseudopotential calculations, we
exploredhowbothalgorithms’time-intensivecomponentsscalewiththenumberofk-pointssam-
15

C
(GTH-QZ)
104
M tensor , a: 2.137
SOS total, a: 2.398
oovv , a: 3.259
MP2 total, a: 2.871
103
102
)s
d
101
n
o
c
e
s
(
e
m
100
iT
10-1
10-2
13 23 33 43 53
N
k
Figure 3: Log scaled time consumption (in seconds) for the M tensor step of the SOS-RILT MP2
algorithm (”M tensor”) , whole of the SOS-RILT MP2 algorithm (”SOS total”), the oovv step of
the conventional MP2 algorithm (”oovv”), and whole of the conventional MP2 algorithms (”MP2
total”), as function of the log scaled number of grid k-points (N ) (1x1x1 to 5x5x5) for Diamond
k
using optimized cc-pVQZ basis set. Each data set includes a trend line, labeled in the legend as
”DatasetName,a=[slopevalue]”.
pled in the Brillouin zone. We specifically analyzed the algorithm’s performance for diamond
usingtheGTH-cc-pVQZbasissetandk-pointgridsrangingfrom N = 1×1×1to N = 5×5×5.
k k
OuranalysisoftheMtensorandtheoovvstep,alongwiththetotaltimeconsumptionforeachal-
gorithm,ispresentedinFig.3. TheresultsindicatethattheMtensorandoovvscaleroughlyclose
to theoretical values (2.137 and 3.259 respectively, compared to theoretical expectations of 2 and
3). Although the scaling of the total time for conventional MP2 was slightly smaller than that of
theoovvstep(2.871vs. 3.259),thescalingofthetotaltimeforSOS-RILT-MP2wasslightlylarger
than that of the M tensor (2.398 vs. 2.137). This discrepancy can be attributed to the scaling of
the M tensor step as N2, while the final part of the algorithm scales as N3 (see Alg. 1). However,
k k
16

Table 1: Comparison of computation times (hours) for the M tensor and the oovv steps, along
with total times for SOS-RILT-MP2 and conventional MP2 algorithms across different basis sets
and k-point grids for the Benzene molecular crystal. Quadrature points for SOS-RILT-MP2 were
chosentoensureµHartreeprecision,with12pointsforDZand15forTZ.
k-pointgrid&cc-pV(XZ)basisset Mtensor SOS-RILT-MP2total oovv MP2total
2x2x2&cc-pVDZ 0.996 1.113 6.732 8.190
2x2x2&cc-pVTZ 3.559 8.077 49.176 66.143
3x3x3&cc-pVDZ 37.568 46.975 275.772 363.655
it is worth mentioning that even with this discrepancy, comparing the total time for the conven-
tional MP2 algorithm in the 5×5×5 case with that of the SOS-RILT-MP2 algorithm, we can see a
cleardifferenceofaboutoneorderofmagnitudeinfavorofSOS-RILT-MP2. Thisanalysisfurther
demonstratestheefficiencyandscalabilityadvantagesofSOS-RILT-MP2overconventionalMP2,
showingsuperiorperformanceinhandlinglargerk-pointgrids,regardlessofthenumberofvirtual
orbitals. This is a huge advantage for reaching the thermodynamic limit and mitigating basis-set
errors,aswellasforcalculatingcomplexunitcellsystemsthatrequirelargek-pointmeshes.
5 Conclusions
In this work, we present a novel SOS-RILT-MP2 algorithm for periodic systems, which reduce
the scaling of the most time-consuming step in conventional MP2 from N3N5 to N2N4, by using
k k
the resolution of the identity approximation combined with the Laplace transform algorithm. We
outline the methodology and algorithmic steps of this approach, utilizing periodic Gaussian basis
sets. Thesebasissetsenablecalculationstobeperformedbothwithandwithoutpseudopotentials.
Wetestedthealgorithmbothonthemolecularsystemswithincreasingthenumberofatomsand
on solids with increasing the basis set size and the number of k-points. In both cases, we showed
that the Laplace transform part of the algorithm can reach micro-Hartree precision with a small
number of quadrature points; the maximal was 19 for diamond with an all-electron, quadruple-
zeta basis set. We verified the efficiency of the algorithm by testing the scaling of the algorithm
with increasing the basis set size and the number of k-points. We show that the SOS-RILT-MP2
17

algorithmhasreducedthescalingofconventionalMP2byanorderofmagnitudewiththenumber
ofatoms(orbasisfunctions)intheunitcell,aswellasreducedscalingwiththenumberofk-points.
We also tested our efficient algorithm on the benzene molecular crystal, achieving a ∼8× speedup
and a saving of 317 hours compared to conventional MP2s. This significantly aids in achieving
the thermodynamic limit and reducing basis-set errors, as well as in calculating complex unit cell
systemsthatnecessitateextensivek-pointmeshes.
PreviousstudiesonperiodicMP2 formolecularcrystalsobservedgoodperformanceforSOS-
MP2 in cohesive energy calculations for some crystals 63 and an underestimation for others such
as benzene .64 In our last work, we applied SOS-MP2 on 12 semiconductors and insulators and
calculated the lattice constant, bulk modulus, and cohesive energies and showed promising results
in various scaling parameters. Future work should focus on applications of the adsorption of
molecules on surfaces for catalysis, defects in materials, and other complex interfaces. A recent
paper calculating the adsorption of the CO molecule on the MgO surface showed that MP2 yields
good results for the adsorption energy and vibrational frequencies.65 Furthermore, extension of
the periodic SOS-RILT algorithm to excited state methods (such as CIS(D) and CC2) should give
greatadvantageincalculatingtheopticalpropertiesofthesematerials.
Data Availability
The input and output data files associated with this study and all analysis are available from the
correspondingauthor,T.G,uponreasonablerequest. ThesourcecodeforperiodicSOS-RILT-MP2
isavailablefromthecorrespondingauthor,T.G,uponreasonablerequest.
Acknowledgement
We thank Timothy Berkelbach for helpful discussions. X.W. acknowledges the start-up funding
from the University of California, Santa Cruz. T.G acknowledges funding from the Ministry of
Innovation,ScienceandTechnologyIsraelgrantNo.5802,andtheMinistryofEnergyIsrael.
18

References
(1) Shavitt, I.; Bartlett, R. J. Many-Body Methods in Chemistry and Physics: MBPT and
Coupled-ClusterTheory;CambridgeMolecularScience;CambridgeUniversityPress: Cam-
bridge,2009.
(2) Bartlett, R. J.; Musiał, M. Coupled-cluster theory in quantum chemistry. Rev. Mod. Phys.
2007,79,291–352.
(3) Hohenberg,P.;Kohn,W.Inhomogeneouselectrongas.Physicalreview1964,136,B864.
(4) Kohn, W.; Sham, L. J. Self-consistent equations including exchange and correlation effects.
Physicalreview1965,140,A1133.
(5) Becke,A.D.AnewmixingofHartree–Fockandlocaldensity-functionaltheories.TheJour-
nalofchemicalphysics1993,98,1372–1377.
(6) Cohen, A. J.; Mori-Sa´nchez, P.; Yang, W. Challenges for Density Functional Theory. Chem.
Rev.2012,112,289–320.
(7) Perdew, J. P. Density functional theory and the band gap problem. International Journal of
QuantumChemistry1985,28,497–523.
(8) Hirata, S.; Podeszwa, R.; Tobita, M.; Bartlett, R. J. Coupled-cluster singles and doubles for
extendedsystems.J.Chem.Phys.2004,120,2581–2592.
(9) Gru¨neis, A.; Booth, G. H.; Marsman, M.; Spencer, J.; Alavi, A.; Kresse, G. Natural Orbitals
for Wave Function Based Correlated Calculations Using a Plane Wave Basis Set. J. Chem.
TheoryComput.2011,7,2780–2785.
(10) Booth, G. H.; Gru¨neis, A.; Kresse, G.; Alavi, A. Towards an Exact Description of Electronic
WavefunctionsinRealSolids.Nature2013,493,365.
19

(11) Gru¨neis, A. A coupled cluster and Møller-Plesset perturbation theory study of the pressure
inducedphasetransitionintheLiHcrystal.J.Chem.Phys.2015,143,102817.
(12) McClain, J.; Sun, Q.; Chan, G. K.-L.; Berkelbach, T. C. Gaussian-Based Coupled-Cluster
Theory for the Ground-State and Band Structure of Solids. J. Chem. Theory Comput. 2017,
13,1209–1218.
(13) Tsatsoulis,T.;Hummel,F.;Usvyat,D.;Schu¨tz,M.;Booth,G.H.;Binnie,S.S.;Gillan,M.J.;
Alfe`, D.; Michaelides, A.; Gru¨neis, A. A comparison between quantum chemistry and quan-
tum Monte Carlo techniques for the adsorption of water on the (001) LiH surface. J. Chem.
Phys.2017,146,204108.
(14) Gruber, T.; Liao, K.; Tsatsoulis, T.; Hummel, F.; Gru¨neis, A. Applying the Coupled-Cluster
AnsatztoSolidsandSurfacesintheThermodynamicLimit.Phys.Rev.X 2018,8,021043.
(15) Gruber, T.; Gru¨neis, A. Ab Initio Calculations of Carbon and Boron Nitride Allotropes and
TheirStructuralPhaseTransitionsUsingPeriodicCoupledClusterTheory.Phys.Rev.B2018,
98,134108.
(16) Zhang, I. Y.; Gru¨neis, A. Coupled Cluster Theory in Materials Science. Front. Mater. 2019,
6,123.
(17) Gao, Y.; Sun, Q.; Yu, J. M.; Motta, M.; McClain, J.; White, A. F.; Minnich, A. J.; Chan, G.
K.-L. Electronic structure of bulk manganese oxide and nickel oxide from coupled cluster
theory.Phys.Rev.B2020,101,165138.
(18) Nusspickel, M.; Booth, G. H. Systematic Improvability in Quantum Embedding for Real
Materials.Phys.Rev.X 2022,12,011046.
(19) Maschio, L.; Usvyat, D.; Manby, F. R.; Casassa, S.; Pisani, C.; Schu¨tz, M. Fast local-
MP2 method with density-fitting for crystals. I. Theory and algorithms. Physical Review
B—CondensedMatterandMaterialsPhysics2007,76,075101.
20

(20) Pisani, C.; Maschio, L.; Casassa, S.; Halo, M.; Schu¨tz, M.; Usvyat, D. Periodic local MP2
method for the study of electronic correlation in crystals: Theory and preliminary applica-
tions.Journalofcomputationalchemistry2008,29,2113–2124.
(21) Mullan, T.; Maschio, L.; Saalfrank, P.; Usvyat, D. Reaction barriers on non-conducting sur-
faces beyond periodic local MP2: Diffusion of hydrogen on α-Al2O3 (0001) as a test case.
TheJournalofChemicalPhysics2022,156.
(22) Møller, C.; Plesset, M. S. Note on an approximation treatment for many-electron systems.
Physicalreview1934,46,618.
(23) Grimme,S.Improvedsecond-orderMøller–Plessetperturbationtheorybyseparatescalingof
parallel-andantiparallel-spinpaircorrelationenergies.J.Chem.Phys.2003,118,9095–9102.
(24) Jung, Y.; Lochan, R. C.; Dutoi, A. D.; Head-Gordon, M. Scaled opposite-spin second or-
der Møller–Plesset correlation energy: An economical electronic structure method. J. Chem.
Phys.2004,121,9793–9802.
(25) Hyla-Kryspin,I.;Grimme,S.Comprehensivestudyofthethermochemistryoffirst-rowtran-
sitionmetalcompoundsbyspincomponentscaledMP2andMP3methods.Organometallics
2004,23,5581–5592.
(26) Takatani, T.; Sherrill, C. D. Performance of spin-component-scaled Møller–Plesset theory
(SCS-MP2)forpotentialenergycurvesofnoncovalentinteractions.Phys.Chem.Chem.Phys.
2007,9,6106–6114.
(27) DistasioJr,R.A.;Head-Gordon,M.Optimizedspin-componentscaledsecond-orderMøller-
Plesset perturbation theory for intermolecular interaction energies. Mol. Phys. 2007, 105,
1073–1083.
(28) Kossmann, S.; Neese, F. Correlated ab initio spin densities for larger molecules: orbital-
optimizedspin-component-scaledMP2method.J.Phys.Chem.A2010,114,11768–11781.
21

(29) Grimme, S.; Goerigk, L.; Fink, R. F. Spin-component-scaled electron correlation methods.
WIREsComput.Mol.Sci.2012,2,886–906.
(30) Grimme, S. Accurate calculation of the heats of formation for large main group compounds
with spin-component scaled MP2 methods. The Journal of Physical Chemistry A 2005, 109,
3067–3077.
(31) Szabados, A´. Theoretical interpretation of Grimme’s spin-component-scaled second order
Møller-Plessettheory.J.Chem.Phys.2006,125,214105.
(32) Grimme,S.;Izgorodina,E.I.Calculationof0–0excitationenergiesoforganicmoleculesby
CIS(D)quantumchemicalmethods.Chemicalphysics2004,305,223–230.
(33) Takatani, T.; Hohenstein, E. G.; Sherrill, C. D. Improvement of the coupled-cluster singles
anddoublesmethodviascalingsame-andopposite-spincomponentsofthedoubleexcitation
correlationenergy.TheJournalofchemicalphysics2008,128.
(34) Hellweg, A.; Gru¨n, S. A.; Ha¨ttig, C. Benchmarking the performance of spin-component
scaledCC2ingroundandelectronicallyexcitedstates.PhysicalChemistryChemicalPhysics
2008,10,4119–4127.
(35) Weigend, F.; Ha¨ser, M. RI-MP2: first derivatives and global consistency. Theoretical Chem-
istryAccounts1997,97,331–340.
(36) Ha¨ser, M.; Almlo¨f, J. Laplace transform techniques in Mo/ller–Plesset perturbation theory.
TheJournalofchemicalphysics1992,96,489–494.
(37) Jurecˇka,P.;Sˇponer,J.;Cˇerny`,J.;Hobza,P.Benchmarkdatabaseofaccurate(MP2andCCSD
(T)completebasissetlimit)interactionenergiesofsmallmodelcomplexes,DNAbasepairs,
andaminoacidpairs.PhysicalChemistryChemicalPhysics2006,8,1985–1993.
(38) Goldzak, T.; Wang, X.; Ye, H.-Z.; Berkelbach, T. C. Accurate thermochemistry of covalent
andionicsolidsfromspin-component-scaledMP2.TheJournalofChemicalPhysics2022,
22

(39) Liang, Y. H.; Ye, H.-Z.; Berkelbach, T. C. Can spin-component scaled MP2 achieve kJ/mol
accuracy for cohesive energies of molecular crystals? The Journal of Physical Chemistry
Letters2023,14,10435–10441.
(40) Dolgonos,G.A.;Hoja,J.;Boese,A.D.RevisedvaluesfortheX23benchmarksetofmolec-
ularcrystals.PhysicalChemistryChemicalPhysics2019,21,24333–24344.
(41) Izmaylov,A.F.;Scuseria,G.E.ResolutionoftheidentityatomicorbitalLaplacetransformed
secondorderMøller–Plessettheoryfornonconductingperiodicsystems.PhysicalChemistry
ChemicalPhysics2008,10,3421–3429.
(42) Ayala, P. Y.; Kudin, K. N.; Scuseria, G. E. Atomic orbital Laplace-transformed second-order
Møller–Plesset theory for periodic systems. The Journal of Chemical Physics 2001, 115,
9698–9707.
(43) Shang, H.; Yang, J. Implementation of laplace transformed mp2 for periodic systems with
numericalatomicorbitals.Frontiersinchemistry2020,8,589992.
(44) Del Ben, M.; Hutter, J.; VandeVondele, J. Electron correlation in the condensed phase from
a resolution of identity approach based on the Gaussian and plane waves scheme. Journal of
chemicaltheoryandcomputation2013,9,2654–2671.
(45) Scha¨fer, T.; Ramberger, B.; Kresse, G. Quartic scaling MP2 for solids: A highly parallelized
algorithmintheplanewavebasis.TheJournalofchemicalphysics2017,146,104101.
(46) Villard, J.; Bircher, M. P.; Rothlisberger, U. Plane waves versus correlation-consistent basis
sets: A comparison of MP2 non-covalent interaction energies in the complete basis set limit.
JournalofChemicalTheoryandComputation2023,19,9211–9227.
(47) Sun, Q.; Berkelbach, T. C.; McClain, J. D.; Chan, G. K.-L. Gaussian and plane-wave mixed
densityfittingforperiodicsystems.J.Chem.Phys.2017,147,164119.
23

(48) Takatsuka, A.; Ten-No, S.; Hackbusch, W. Minimax approximation for the decomposition
of energy denominators in Laplace-transformed Møller–Plesset perturbation theories. The
Journalofchemicalphysics2008,129.
(49) Almlo¨f, J. Elimination of energy denominators in Møller—Plesset perturbation theory by a
Laplacetransformapproach.Chemicalphysicsletters1991,181,319–320.
(50) Hackbusch, W. Computation of Best L∞ Exponential Sums for 1/x by Remez’ Algorithm.
ComputingandVisualizationinScience2019,20,1–11.
(51) Hackbusch, W. Best L∞ Exponential Sums for 1/x. https://gitlab.mis.mpg.de/
scicomp/EXP_SUM/-/tree/main/1_x,Accessed: 2025-02-17.
(52) Sun,Q.;Berkelbach,T.C.;Blunt,N.S.;Booth,G.H.;Guo,S.;Li,Z.;Liu,J.;McClain,J.D.;
Sayfutyarova, E. R.; Sharma, S.; others PySCF: the Python-based simulations of chem-
istry framework. Wiley Interdisciplinary Reviews: Computational Molecular Science 2018,
8,e1340.
(53) Sun, Q.; Zhang, X.; Banerjee, S.; Bao, P.; Barbry, M.; Blunt, N. S.; Bogdanov, N. A.;
Booth, G. H.; Chen, J.; Cui, Z.-H.; Eriksen, J. J.; Gao, Y.; Guo, S.; Hermann, J.; Her-
mes, M. R.; Koh, K.; Koval, P.; Lehtola, S.; Li, Z.; Liu, J.; Mardirossian, N.; McClain, J. D.;
Motta, M.; Mussard, B.; Pham, H. Q.; Pulkin, A.; Purwanto, W.; Robinson, P. J.; Ronca, E.;
Sayfutyarova, E. R.; Scheurer, M.; Schurkus, H. F.; Smith, J. E. T.; Sun, C.; Sun, S.-N.;
Upadhyay, S.; Wagner, L. K.; Wang, X.; White, A.; Whitfield, J. D.; Williamson, M. J.;
Wouters, S.; Yang, J.; Yu, J. M.; Zhu*, T.; Berkelbach, T. C.; Sharma, S.; Sokolov*, A. Y.;
Chan, G. K.-L. Recent developments in the PySCF program package. J. Chem. Phys. 2020,
153,024109.
(54) Paier, J.; Hirschl, R.; Marsman, M.; Kresse, G. The Perdew–Burke–Ernzerhof exchange-
correlation functional applied to the G2-1 test set using a plane-wave basis set. The Journal
ofchemicalphysics2005,122.
24

(55) Broqvist, P.; Alkauskas, A.; Pasquarello, A. Hybrid-functional calculations with plane-wave
basis sets: Effect of singularity correction on total energies, energy eigenvalues, and de-
fect energy levels. Physical Review B—Condensed Matter and Materials Physics 2009, 80,
085114.
(56) Sundararaman, R.; Arias, T. Regularization of the Coulomb singularity in exact exchange
by Wigner-Seitz truncated interactions: Towards chemical accuracy in nontrivial systems.
PhysicalReviewB—CondensedMatterandMaterialsPhysics2013,87,165122.
(57) Goedecker, S.; Teter, M.; Hutter, J. Separable Dual-Space Gaussian Pseudopotentials. Phys.
Rev.B1996,54,1703–1710.
(58) Hartwigsen, C.; Goedecker, S.; Hutter, J. Relativistic Separable Dual-Space Gaussian Pseu-
dopotentialsfromHtoRn.Phys.Rev.B1998,58,3641–3662.
(59) Ye, H.-Z.; Berkelbach, T. C. Correlation-Consistent Gaussian Basis Sets for Solids Made
Simple.J.Chem.TheoryComput.2022,18,1595–1606.
(60) Dunning Jr, T. H. Gaussian basis sets for use in correlated molecular calculations. I. The
atoms boron through neon and hydrogen. The Journal of chemical physics 1989, 90, 1007–
1023.
(61) Gaurav, H.; Vibin, A.; Zgid, D. Challenges with relativistic GW calculations in solids and
molecules.FaradayDiscussions2024,
(62) Zhu,T.;Chan,G.K.-L.All-electronGaussian-basedG0W0forvalenceandcoreexcitation
energies of periodic systems. Journal of Chemical Theory and Computation 2021, 17, 727–
741.
(63) DelBen,M.;Hutter,J.;VandeVondele,J.Second-orderMøller–Plessetperturbationtheoryin
thecondensedphase: AnefficientandmassivelyparallelGaussianandplanewavesapproach.
Journalofchemicaltheoryandcomputation2012,8,4177–4188.
25

(64) Bintrim, S. J.; Berkelbach, T. C.; Ye, H.-Z. Integral-Direct Hartree–Fock and Møller–Plesset
Perturbation Theory for Periodic Systems with Density Fitting: Application to the Benzene
Crystal.JournalofChemicalTheoryandComputation2022,18,5374–5381.
(65) Ye, H.-Z.; Berkelbach, T. C. Adsorption and Vibrational Spectroscopy of CO on the Surface
ofMgOfromPeriodicLocalCoupled-ClusterTheory.FaradayDiscussions2024,
26
