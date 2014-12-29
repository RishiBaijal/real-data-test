from __future__ import print_function, division
import numpy as np
import pandas as pd
from os import *
from os.path import *

import nilmtk
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from nilmtk.disaggregate import CombinatorialOptimisation
from nilmtk.utils import print_dict
from nilmtk.metrics import f1_score

from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilmtk.utils import check_directory_exists
from nilm_metadata import *
from inspect import currentframe, getfile, getsourcefile
from sys import getfilesystemencoding
import cProfile
import time

import warnings
warnings.filterwarnings("ignore")

import logging


def list_of_appliances(elec):
    lis=[]
    for i, obj in enumerate(elec.meters):
        if (len(obj.appliances) > 0):
            lis.append (obj.appliances[0].type['type'])
        if (len(obj.appliances)==0):
            lis.append(None)
    return list(set(lis))





def test_all_datasets(directory):
    print("Testing all data sets started at:  {}".format(time.time()))
    print("="*60)
    check_directory_exists(directory)
    datasets = [f for f in listdir(directory) if isfile(join(directory, f)) and
         '.h5' in f and '.swp' not in f]
    for dataset in datasets:
        dataset_obj=DataSet(join(directory, dataset))
        test_single_dataset(dataset_obj)


def test_single_dataset(dataset_obj):
    print("Testing all function of {} dataset started at:  {}".format(dataset_obj, time.time()))
    print("="*60)
    test_all_buildings(dataset_obj)
    test_metadata_dataset(dataset_obj)

def test_all_buildings(dataset):
    buildings = dataset.buildings
    print (buildings)
    for building in buildings:
        test_single_building(buildings[building])

def test_single_building(building):
    print ("The type of building is ", type(building))
    test_single_building_metadata(building)
    elec = building.elec
    test_single_meter_group(elec)


def test_single_building_metadata(building):
    try:
        print(building.metadata)
    except Exception, e:
        # log it..
        logging.exception(e)

    try:
        print (building.identifier)
    except:
        logging.exception(e)
    


    try:
        print(building.describe())
    except Exception, e:
        # log it
        logging.exception(e)




def test_single_meter_group(elec):
    try:
        print ("Meters: ")
        for meter in elec.meters:
            print (meter)
        print ("-"*60)
        print ("Appliances: ")
        for appliance in elec.appliances:
            print (appliance)
        print ("-"*60)
        print ("Good sections: ")
        elec.good_sections()
        print ("-"*60)
        print  ("Checking if site meter", elec.is_site_meter())
        print ("-"*60)
        print ("Printing site meters ", elec.mains())
        print ("-"*60)
        print ("Identifier ")
        print (elec.identifier)
        print ("-"*60)
        print ("Tuple of meter instances: ")
        print (elec.instance())
        print ("-"*60)
        print ("Creating generator load object.")
        elec.load()
        print ("-"*60)
        print ("Printing the meters directly downstream of mains.")
        elec.meters_directly_downstream_of_mains()
        print ("-"*60)
        print ("Printing nested metergroups", elec.nested_metergroups())
        print ("-"*60)
        print ("Timeframe: ", elec.get_timeframe())
        print ("-"*60)
        print ("Available power AC types: ", elec.available_power_ac_types())
        print ("-"*60)
        print ("Clearing cache...done.")
        elec.clear_cache()
        print ("-"*60)
        print ("Testing if there are meters from multiple buildings. Result returned by method: ", elec.contains_meters_from_multiple_buildings())
        print ("-"*60)
        print ("List of disabled meters: ", elec.disabled_meters)
        print ("-"*60)
        print ("Power series: ")
	elec.power_series()
	elec.power_series_all_data()
	print ("-"*60)
	print ("Printing sub-meters: ", elec.submeters())
	print ("-"*60)
	print ("Testing switch_times: ", elec.switch_times())
	print ("-"*60)
	print ("Total energy: ", elec.total_energy())
	print ("-"*60)
	#print ("Computing pairwise correlation. This will take some time...", elec.pairwise_correlation())
	print ("-"*60)
	print ("Computing uptime: ", elec.uptime())
	print ("-"*60)
	print (elec.use_alternative_means())
	print ("-"*60)
	print ("Vampire power: ", elec.vampire_power())
	print ("-"*60)
	elec.when_on()
	
        print ("Trying to determine the dominant appliance: ")
        try:
            elec.dominant_appliance()
        except RuntimeError:
            print ('''More than one dominant appliance in MeterGroup! (The dominant appliance per meter should be manually specified in the metadata. If it isn't and if there are multiple appliances for a meter then NILMTK assumes all appliances on that meter are dominant. NILMTK can't automatically distinguish between multiple appliances on the same meter (at least, not without using NILM!))''')
            pass
                #print ("Dropout rate: ", elec.dropout_rate())
        print ("-"*60)
        try:
            print ("Calculating energy per meter:")
            print (elec.energy_per_meter())
            print ("Calculating total entropy")
            print (elec.entropy())
            
            print ("Calculating entropy per meter: ")
            print (elec.entropy_per_meter())
            print ("Calculating fraction per meter: ")
            print (elec.fraction_per_meter())
        except ValueError:
            print ("ValueError: Total size of array must remain unchanged.")
            pass
        print ("-"*60)
        print ("Calculating fraction per meter.")
        elec.clear_cache()
        elec.fraction_per_meter()
        print (elec.fraction_per_meter())
        print ("-"*60)

    except Exception as e:
        logging.exception(e)
        pass

    return None




def test_metadata_dataset(dataset):
    try:
        print(dataset.metadata)
    except Exception as e:
        logging.exception(e)
        # Do something....maybe log it somewhere

def test_all_elecmeters(metergroup):
    return None

def test_elecmeter(elecmeter):
    return None



def test_all_meter_groups():
    return None

test_all_datasets('/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/NILMTK_datasets')


#def test_single_mete


"""

    files = [f for f in listdir(path_to_directory) if isfile(join(path_to_directory, f)) and
         '.h5' in f and '.swp' not in f]
    files.sort()

    print ("Datasets collected and sorted. Processing...")


    try:
        for i, file in enumerate(files):
            current_file=DataSet(join(path_to_directory, file))

            print ("Printing metadata for current file...done.")
            print_dict(current_file.metadata)
            print (" Loading file # ", i, " : ", file, ". Please wait.")
            for building_number in range(1, len(current_file.buildings)+1):
    #Examine metadata for a single house
                elec=current_file.buildings[building_number].elec
                print ("The dataset being processed is : ", elec.dataset())
                print ("Metadata for current file: ")
                print_dict(current_file.buildings[building_number].metadata)
                print ("Appliance label information: ", elec.appliance_label())
                #print (elec.appliances)
                print ("Appliances:- ")
                for i in elec.appliances:
                    print (i)

                print ("Examining sub-metered appliances...")


                print ("Collecting stats on meters...Done.")
                print (elec._collect_stats_on_all_meters)

                print ("Timeframe: ", elec.get_timeframe())




                print ("Available power AC types: ", elec.available_power_ac_types())

                print ("Clearing cache...done.")
                elec.clear_cache()

                print ("Testing if there are meters from multiple buildings. Result returned by method: ", elec.contains_meters_from_multiple_buildings())

                # TODO: Find a better way to test the correlation function
                # print ("Testing the correlation function. ", elec.correlation(elec))


                print ("List of disabled meters: ", elec.disabled_meters)
                print ("Trying to determine the dominant appliance: ")
                try:
                    elec.dominant_appliance()
                except RuntimeError:
                    print ('''More than one dominant appliance in MeterGroup! (The dominant appliance per meter should be manually specified in the metadata. If it isn't and if there are multiple appliances for a meter then NILMTK assumes all appliances on that meter are dominant. NILMTK can't automatically distinguish between multiple appliances on the same meter (at least, not without using NILM!))''')
                    pass
                print ("Dropout rate: ", elec.dropout_rate())
                try:
                    print ("Calculating energy per meter:")
                    print (elec.energy_per_meter())

                    print ("Calculating total entropy")
                    print (elec.entropy())

                    print ("Calculating entropy per meter: ")
                    print (elec.entropy_per_meter())
                except ValueError:
                    print ("ValueError: Total size of array must remain unchanged.")
                    pass

                print ("Calculating fraction per meter.")
                print (elec.fraction_per_meter())




#print ("Average energy per period: ", elec.average_energy_per_period())


                print ("Executing functions...")
                lis=[]
                func=""
                '''for function in dir(elec):
                    try:
                        start=time.time()
                        if ("__" not in function or "dataframe_of_meters" not in function):
                            func=getattr(elec, function)
                        print ("Currently executing ", function, ". Please wait...")
                        print (func())
                        # print ("cProfile stats - printed")
                        # cProfile.run("func")
                        end=time.time()
                        print ("Time taken for the entire process : ", (end - start))
                    except AttributeError:
                        print ("Attribute error occured. ")
                    except TypeError:
                        lis.append(function)
                        print ("Warning: TypeError")
                        pass'''

                print ("Plotting wiring hierarchy of meters....")
                elec.draw_wiring_graph()
                ## DISAGGREGATION STARTS HERE
                appliance_type="unknown"
    #TODO : appliance_type should cycle through all appliances and check for each of them. For this, use a list.
                selected_appliance=nilmtk.global_meter_group.select_using_appliances(type=appliance_type)
                appliance_restricted = MeterGroup(selected_appliance.meters)
                if ((appliance_restricted.proportion_of_upstream_total_per_meter()) is not None):
                    proportion_per_appliance = appliance_restricted.proportion_of_upstream_total_per_meter()


                    proportion_per_appliance.plot(kind='bar');
                    plt.title('Appliance energy as proportion of total building energy');
                    plt.ylabel('Proportion');
                    plt.xlabel('Appliance (<appliance instance>, <building instance>, <dataset name>)');
                    selected_appliance.select(building=building_number).total_energy()
                    selected_appliance.select(building=1).plot();


                    appliance_restricted = MeterGroup(selected_appliance.meters)
                    daily_energy = pd.DataFrame([meter.average_energy_per_period(offset_alias='D')
                                     for meter in appliance_restricted.meters])

                    daily_energy.plot(kind='hist');
                    plt.title('Histogram of daily energy');
                    plt.xlabel('energy (kWh)');
                    plt.ylabel('Occurences');
                    plt.legend().set_visible(False)

                    current_file.store.window=TimeFrame(start='2012-04-01 00:00:00-05:00', end='2012-04-02 00:00:00-05:00')
                    #elec.plot();

                    fraction = elec.submeters().fraction_per_meter().dropna()

                    labels = elec.get_appliance_labels(fraction.index)
                    plt.figure(figsize=(8,8))
                    fraction.plot(kind='pie', labels=labels);

                    elec.select_using_appliances(category='heating')
                    elec.select_using_appliances(category='single-phase induction motor')


                    co = CombinatorialOptimisation()
                    co.train(elec)

                    for model in co.model:
                        print_dict(model)


                    disag_filename = join(data_dir, 'ampds-disag.h5')
                    output = HDFDataStore(disag_filename, 'w')
                    co.disaggregate(elec.mains(), output)
                    output.close()



                    disag = DataSet(disag_filename)








                    disag_elec = disag.buildings[building_number].elec

                    f1 = f1_score(disag_elec, elec)
                    f1.index = disag_elec.get_appliance_labels(f1.index)
                    f1.plot(kind='bar')
                    plt.xlabel('appliance');
                    plt.ylabel('f-score');
                    disag_elec.plot()

                    disag.store.close()
    except AttributeError:
        print ("AttributeError occured while executing. This means that the value returned by  proportion_per_appliance = appliance_restricted.proportion_of_upstream_total_per_meter() is None")
        pass






test_all('/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/NILMTK_datasets')
#test_all('/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/nilmtk/data/Location_of_h5')

"""
