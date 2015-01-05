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
    lis = []
    for i, obj in enumerate(elec.meters):
        if (len(obj.appliances) > 0):
            lis.append(obj.appliances[0].type['type'])
        if (len(obj.appliances) == 0):
            lis.append(None)
    return list(set(lis))


def test_all_datasets(directory):
    print("Testing all data sets started at:  {}".format(time.time()))
    print("=" * 60)
    check_directory_exists(directory)
    datasets = [f for f in listdir(directory) if isfile(join(directory, f)) and
                '.h5' in f and '.swp' not in f]
    for dataset in datasets:
        dataset_obj = DataSet(join(directory, dataset))
        test_single_dataset(dataset_obj)


def test_single_dataset(dataset_obj):
    print("Testing all functions of {} dataset started at:  {}".format(
        dataset_obj, time.time()))
    print("=" * 60)
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
    passed=0
    error=0
    try:
        print ("Meters: ")
        for meter in elec.meters:
            print (meter)
        print ("-" * 60)
        print ("Appliances: ")
        for appliance in elec.appliances:
            print (appliance)
        print ("-" * 60)
        print ("Good sections: ")
        try:
            elec.good_sections()
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        try:
            print ("Checking if site meter", elec.is_site_meter())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        try:
            print ("Printing site meters ", elec.mains())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        print ("Identifier ")
        print (elec.identifier)
        print ("-" * 60)
        print ("Tuple of meter instances: ")
        try:
            print (elec.instance())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        print ("Creating generator load object.")
        try:
            elec.load()
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        try:
            print ("Printing the meters directly downstream of mains.",
                   elec.meters_directly_downstream_of_mains())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        try:
            print ("Printing nested metergroups", elec.nested_metergroups())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)

        try:
            print ("Timeframe: ", elec.get_timeframe())
        except Exception, e:
            logging.exception(e)
            error+=1
            pass

        print ("-" * 60)
        try:
            print ("Available power AC types: ", elec.available_power_ac_types())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        print ("Clearing cache...done.")
        try:
            elec.clear_cache()
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        try:
            print ("Testing if there are meters from multiple buildings. Result returned by method: ",
               elec.contains_meters_from_multiple_buildings())
        except Exception as e:
            logging.exception (e)
	    error+=1
            pass
        print ("-" * 60)
        try:
            print ("List of disabled meters: ", elec.disabled_meters)
        except Exception as e:
            logging.exception(e)
            error+=1
            pass

        print ("-" * 60)
        print ("Power series: ")
        try:
            elec.power_series()
            elec.power_series_all_data()
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        print ("Printing sub-meters: ", elec.submeters())
        print ("-" * 60)
        try:
            print ("Testing switch_times: ", elec.switch_times())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        try:
            print ("Total energy: ", elec.total_energy())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        #print ("Computing pairwise correlation. This will take some time...", elec.pairwise_correlation())
        print ("-" * 60)
        try:
            print ("Computing uptime: ", elec.uptime())
        except Exception as e:
            logging.exception(e)
	    error+=1
            pass
        print ("-" * 60)
        try:
            print (elec.use_alternative_mains())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        try:
            print ("Vampire power: ", elec.vampire_power())
        except Exception as e:
            logging.exception(e)
            error+=1
            pass
        print ("-" * 60)
        try:
            elec.when_on()
        except Exception as e:
            logging.exception(e)
            error+=1
            pass

        print ("Trying to determine the dominant appliance: ")
        try:
            elec.dominant_appliance()
        except RuntimeError:
            print (
                '''More than one dominant appliance in MeterGroup! (The dominant appliance per meter should be manually specified in the metadata. If it isn't and if there are multiple appliances for a meter then NILMTK assumes all appliances on that meter are dominant. NILMTK can't automatically distinguish between multiple appliances on the same meter (at least, not without using NILM!))''')
            error+=1
            pass
            #print ("Dropout rate: ", elec.dropout_rate())
        print ("-" * 60)
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
            error+=1
            pass
        print ("-" * 60)
        print ("Calculating fraction per meter.")
        elec.clear_cache()
        elec.fraction_per_meter()
        print (elec.fraction_per_meter())
        print ("-" * 60)

    except Exception as e:
        logging.exception(e)
        pass

    print ("Number of functions tested = ", len(dir(elec)))
    print ("Number of functions passed = ", len(dir(elec)) - error)
    print ("Number of errors encountered = ", error)
    
    return None


def test_metadata_dataset(dataset):
    try:
        print(dataset.metadata)
    except Exception as e:
        logging.exception(e)


test_all_datasets(
    '/Users/rishi/Documents/Master_folder/IIITD/5th_semester/Independent_Project/NILMTK_datasets')


# def test_single_mete

