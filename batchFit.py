from sedfitter.fit import Fitter
from sedfitter.extinction import Extinction
from sedfitter.source import Source
import numpy as np
import pandas as pd
import astropy.units as u
import argparse
from itertools import islice

parser = argparse.ArgumentParser()
parser.add_argument('startindex', type=int, help='Index of the first source to be fit')
parser.add_argument('endindex', type=int, help='Index of the last source to be fit INCLUSIVE')
args = parser.parse_args()

startindex = args.startindex
endindex = args.endindex

extinction = Extinction.from_file('../data/raw/whitney.r550.par')
apertures = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 7.6] * u.arcsec
filters = ['2J', '2H', '2K', 'I1', 'I2', 'I3', 'I4', 'M1']
filterpeaks = [1.235, 1.662, 2.159, 3.6, 4.5, 5.8, 8.0, 23.675]

sourcepath = '../data/raw/SESNA_normalized' # Want to split this source path into a specified chunk of sources
sourcefile = open(sourcepath,'r')
savables = pd.DataFrame(columns = ['ID', 'Model', 'Model Fluxes', 'Source Flux', 'Valid', 'Chi^2', 'Chi^2 DOF', 'Av', 'Scale'])

numfits = endindex - startindex + 1 # Think 0 to 10 is 11 fits if we include both 0 and 10
index = startindex

for line in islice(sourcefile,startindex,endindex+1): # This for loop will run through each Source in specified range of SESNA
    try: s = Source.from_ascii(line)
    except EOFError: break

    fitter = Fitter(filters, apertures, '../data/galaxtemps',
            extinction_law=extinction,
            distance_range=[0.8, 2] * u.kpc,
            av_range=[0, 100.], remove_resolved=True)
    
    info = fitter.fit(s)
    if ((index-startindex) % 5) == 0:
        print("Fitting source at index: %3.0f " % index)
    # Data to be saved (modelfluxes, chi2, chi2 deg of freedom,)
    modelflux = info.model_fluxes
    chi2 = info.chi2
    chi2_DOF = len([x for x in s.valid if x == 1])
    sourcename = info.source.name
    source = np.array(info.source.to_ascii().split()[11:-1:2]).astype(float) # Make sure this is pulling the correct values. We take 11 to the last value because first 3 indices are ID and ra/dec, followed by 8 valid numbers
    #sourceerror = np.array(info.source.to_ascii().split()[12::2]).astype(float) # This grabs error
    mask = np.array([True if x == 1 else False for x in info.source.valid])
    fitav = info.av
    fitsc = info.sc
    modelname = info.model_name
    savables.loc[len(savables.index)] = [sourcename, modelname, modelflux, source, mask, chi2, chi2_DOF, fitav, fitsc]

    if index < endindex: # We check exclusively equal to because we check after the fit happens, so the last fit when index = endindex has already happened
        index += 1
    else:
        break

savables.to_csv('../data/processed/SESNAFITS_'+str(args.startindex)+'_to_'+str(args.endindex)+'.csv')


sourcefile.close()