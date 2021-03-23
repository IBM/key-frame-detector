# key-frame-detector
Detection of keyframes (i.e. scans with AMD pathologies) in Bioptigen macular scans. 

# Abstract
Spectral-domain optical coherence tomography (SDOCT) is a non-invasive imaging modal- ity that generates high-resolution volumetric images. This modality finds widespread usage in ophthalmology for the diagnosis and management of various ocular conditions. The vol- umes generated can contain 200 or more B-scans. Manual inspection of such large quantity of scans is time consuming and error prone in most clinical settings. Here, we present a method for the generation of visual summaries of SDOCT volumes, wherein a small set of B-scans that highlight the most clinically relevant features in a volume are extracted. The method was trained and evaluated on data acquired from age-related macular degeneration patients, and “relevance” was defined as the presence of visibly discernible structural abnor- malities. The summarisation system consists of a detection module, where relevant B- scans are extracted from the volume, and a set of rules that determines which B-scans are included in the visual summary. Two deep learning approaches are presented and com- pared for the classification of B-scans—transfer learning and de novo learning. Both approaches performed comparably with AUCs of 0.97 and 0.96, respectively, obtained on an independent test set. The de novo network, however, was 98% smaller than the transfer learning approach, and had a run-time that was also significantly shorter.

## Data used for training
Macular scans from Sina Farsiu (Duke University) http://people.duke.edu/~sf59/RPEDC_Ophth_2013_dataset.htm
Citation: S. Farsiu, SJ. Chiu, R.V. O’Connell, F.A. Folgar, E. Yuan, J.A. Izatt, and C.A. Toth, "Quantitative Classification of Eyes with and without Intermediate Age-related Macular Degeneration Using Optical Coherence Tomography", Ophthalmology, 121(1),  162-172 Jan. (2014).

## Results 
see here Antony BJ, Maetschke S, Garnavi R (2019), "Automated summarisation of SDOCT volumes using deep learning: Transfer learning vs de novo trained networks." PLOS ONE 14(5): e0203726. https://doi.org/10.1371/journal.pone.0203726