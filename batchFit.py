# Fitting in 10 batches: 0 - 564558, 564559 - 1129117, 1129118 - 1693676, 1693677 - 2258235, 2258236 - 2822794, 
# 2822795 - 3387353, 3387354 - 3951912, 3951913 - 4516471, 4516472 - 5081030, 5081031 - 5645589
# python batchFit.py 0 564558


from sedfitter.fit import Fitter
from sedfitter.extinction import Extinction
from sedfitter.source import Source
import numpy as np
import pandas as pd
import astropy.units as u
import argparse
from itertools import islice

print("Starting batchFit.py at time: " + str(pd.Timestamp.now()))

# Argparse allows us to give the start and end index (inclusive) as command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('startindex', type=int, help='Index of the first source to be fit')
parser.add_argument('endindex', type=int, help='Index of the last source to be fit INCLUSIVE')
args = parser.parse_args()

startindex = args.startindex
endindex = args.endindex

extinction = Extinction.from_file('../data/raw/whitney.r550.par') # Important our extinction profile
apertures = [2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4, 7.6] * u.arcsec
filters = ['2J', '2H', '2K', 'I1', 'I2', 'I3', 'I4', 'M1']
filterpeaks = [1.235, 1.662, 2.159, 3.6, 4.5, 5.8, 8.0, 23.675]

sourcepath = '../data/raw/SESNA_normalized'
sourcefile = open(sourcepath,'r')

rowlist = [] # Row list will hold all source fit information until the concat command after the loop. This speeds up the fit by not growing a dataframe

numfits = endindex - startindex + 1 # Think 0 to 10 is 11 fits if we include both 0 and 10
index = startindex

sourcecatalog = pd.read_pickle('../data/raw/SESNA_INPUTS_SourceID-Catalog_JakeJuly18.pkl')
sourcecatalog = sourcecatalog.iloc[startindex:endindex+1]
sourcecatalog.set_index('ID',inplace=True) # We pull the source catalog from our list of catalogs in order to get the correct distance

for line in islice(sourcefile,startindex,endindex+1): # This for loop will run through each Source in specified range of SESNA
    try: s = Source.from_ascii(line) # Instantiate Source object
    except EOFError: break

    distances = sourcecatalog.loc[s.name][['DistMin','DistMax']].values

    fitter = Fitter(filters, apertures, '../data/galaxtemps2', # Call Robitaille's fit routine by instantiating a Fitter object
            extinction_law=extinction,
            distance_range=distances * u.kpc,
            av_range=[0, 100.], remove_resolved=True)
    
    info = fitter.fit(s) # Call the fit routine on that fitter using the source object
    info.keep(('N',10)) # Keep top 10 fits
    if ((index-startindex) % 100) == 0:
        time = pd.Timestamp.now()
        print(f"Fitting source at index:{index:10.0f} at time: {time}")
    # Data to be saved (modelfluxes, chi2, chi2 deg of freedom,)
    modelflux = info.model_fluxes
    chi2 = info.chi2
    chi2_DOF = len([x for x in s.valid if x == 1])
    sourcename = info.source.name
    source = np.array(info.source.to_ascii().split()[11:-1:2]).astype(float) # Make sure this is pulling the correct values. We take 11 to the last value because first 3 indices are ID and ra/dec, followed by 8 valid numbers
    mask = np.array([True if x == 1 else False for x in info.source.valid])
    fitav = info.av
    fitsc = info.sc
    modelname = info.model_name

    row = pd.DataFrame(columns = ['ID', 'Model', 'Model Fluxes', 'Source Flux', 'Valid', 'Chi^2', 'Chi^2 DOF', 'Av', 'Scale'])
    row.loc[len(row.index)] = [sourcename, modelname, modelflux, source, mask, chi2, chi2_DOF, fitav, fitsc]
    rowlist.append(row) # Append single row dataframe to the list

    if index < endindex: # We check exclusively less than because we check after the fit happens, so the last fit when index = endindex has already happened
        index += 1
    else:
        break

savables = pd.concat(rowlist,ignore_index=True) # Concat all rows into a single dataframe
savables.to_pickle('../data/processed/Second_Models_Fnu/SESNAFITS2_'+str(args.startindex)+'_to_'+str(args.endindex)+'.pkl')
# Save to pickle format

sourcefile.close()