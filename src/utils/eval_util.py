# Utility functions supporting evaluation
from datetime import datetime, timedelta
import os
import pandas as pd
from src.utils.models_util import get_selected_submodel_name
from src.utils.general_util import string_to_dt, printf
import calendar

def mean_rmse_to_score(mean_rmse):
    """Returns frii contest score associated with a mean RMSE value
    """
    return 100 / (0.1 * mean_rmse + 1)


def score_to_mean_rmse(score):
    """Returns mean RMSE value associated with an frii contest score
    """
    return ((100 / score) - 1) / 0.1


def first_day_of_week(year, day_of_week):
    """Returns the datetime of the first Monday, Tuesday, ..., or Sunday in a given
    year.

    Args:
      year: integer representing year
      day_of_week: string specifying target day of the week in
        {"Monday", ..., "Sunday"}
    """
    # Map day name to Pandas integer
    names_to_ints = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    dt = datetime(year=year, month=1, day=1)
    return dt + timedelta(days=(names_to_ints[day_of_week] - dt.weekday()) % 7)

def get_named_targets():
    """ Return a list of named target date ranges """
    return ["std_train", "std_val", "std_test", "std_ens", "std_all", \
        "std_future", "std_contest_fri", "std_contest", "std_contest_daily", "std_contest_eval", \
        "std_contest_eval_daily", "std_paper", "std_paper_daily"]

def get_target_dates(date_str="std_test", horizon=None):
    """Return list of target date datetime objects for model evaluation.
        Note: returned object should always be a list, even for single target dates

    Args
    ----------
    date_str : string
        Either a named set of target dates (
        "std_train" for model training,
        "std_val" for hyperparameter tuning,
        "std_test" for final model comparison,
        "std_ens" for training ensemble,
        "std_all" for all dates in std_train, std_ens and std_val,
        "std_future" for dates after all other sets
        "std_contest_fri" for weekly rodeo 1 contest dates
        "std_contest" for weekly contest dates
        "std_contest_daily" for daily dates during the contest period
        "std_contest_eval" for biweekly contest dates during the contest period for 2010-2018
        "std_contest_eval_daily" for daily contest dates during the contest period for 2010-2018
        "std_paper" for biweekly Wednesdays for the period Jan 2011-2020, for paper performance metrics 
        "std_paper_eval" daily dates for the period Jan 2007-2020, for model tuning for paper,
        an inclusive date range of the form '19990101-20201231',
        or a string with comma-separated dates in YYYYMMDD format
        (e.g., '20170101,20170102,20180309'))
    horizon: string, Either "34w" or "56w". Used for generated contest date periods. 34w horizons
        exclude the final target date, 56w exclude the initial target date. If no horizon is passed
        will raise a warning and produde the union of the two horizon target dates.

    Returns
    -------
    list
        List with datetime objects
    """
    if os.path.exists(date_str):
        dates=[]
        with open(date_str,'r') as f:
            for line in f.readlines():
                dates.append(datetime.strftime(line.rstrip(),"%Y%m%d"))
        return dates
    
    if date_str == "std_train":
        first_year = 1948
        number_of_days = (365 * 47) + (366 * 16)
    elif date_str == "std_test":
        first_year = 2017
        number_of_days = 365 * 3
    elif date_str == "std_val":
        first_year = 2014
        number_of_days = 365 * 2 + 366
    elif date_str == "std_ens":
        first_year = 2011
        number_of_days = 365 * 2 + 366
    elif date_str == "std_all":
        first_year = 1948
        number_of_days = ((365 * 47) + (366 * 16)) + (365 * 2 + 366) + (365 * 2 + 366)
    elif date_str == "std_future":
        first_year = 2020
        number_of_days = 366
    elif date_str == "std_contest_fri":
        '''
        Standard contest dates for rodeo 1, Tuesdays for 18 Apr 2017 - 03 Apr 2018. If a horizon is passed,
        correctly accounts for missing period at end/start of period for 34w/56w respectively.
        '''
        cstart = datetime(year=2017, month=5, day=2) if horizon == '34w' else datetime(year=2017, month=5, day=16)
        cend = datetime(year=2018, month=4, day=17) if horizon == '56w' else datetime(year=2018, month=5, day=1)
        dates = [cstart + timedelta(days=x) for x in range(0, 364, 14)]
        return dates
    elif date_str == "std_contest":
        '''
        Standard contest dates, Tuesdays for 29 Oct 2019 - 27 Oct 2020. If a horizon is passed,
        correctly accounts for missing period at end/start of period for 34w/56w respectively.
        '''
        cstart, cend = contest_start_end(horizon, year=2019, dow=1)
        dates = [cstart + timedelta(days=x) for x in range(0, 364, 14)]
        return dates

    elif date_str == "std_contest_daily":
        '''
        Daily dates during contest period 29 Oct 2019 - 09 Nov 2020. If a horizon is passed,
        correctly accounts for missing period at end/start of period for 34w/56w respectively.
        '''
        cstart, cend = contest_start_end(horizon, year=2019, dow=1)
        dates = [cstart + timedelta(days=x) for x in range(0, 364)]
        return dates

    elif date_str == "std_contest_eval":
        '''
        Standard contest dates for a multiyear period in the past 2010-2018. A contest period for
        year yyyy is defined as 26 predictions tarting from the last Wednesday in October in year yyyy.
        If a horizon is passed, correctly accounts for missing period at end/start of period
        for 34w/56w respectively.
        '''
        multiyear_dates = []
        for y in range(2010, 2020):
            cstart, cend = contest_start_end(horizon, y, dow=2) # Wednesday
            dates = [cstart + timedelta(days=x) for x in range(0, 365, 14) if cstart + timedelta(days=x) <= cend]
            multiyear_dates += dates

        # Remove duplicates (TODO: question, will there ever be duplicates?)
        # return sorted(set(multiyear_dates), key=lambda x: multiyear_dates.index(x)) # Could improve efficiency 
        return multiyear_dates

    elif date_str == "std_contest_eval_daily":
        ''' 
        Daily dates during a multiyear contest period in the past. If a horizon is passed, 
        correctly accounts for missing period at end/start of period for 34w/56w respectively.
        '''
        multiyear_dates = []
        for y in range(2010, 2020):
            cstart, cend = contest_start_end(horizon, year=y, dow=2) # Wednesday
            dates = [cstart + timedelta(days=x) for x in range(0, 364)] 
            multiyear_dates += dates

        # Remove duplicates (TODO: question, will there ever be duplicates?)
        # return sorted(set(multiyear_dates), key=lambda x: multiyear_dates.index(x)) # Could improve efficiency 
        return multiyear_dates

    elif date_str == "std_paper":
        ''' 
        Paper performance evaluation period from 2011-2020. A year period for 
        year yyyy is defined as 52 predictions weekly starting from the first Wednesday in January 
        '''
        multiyear_dates = []
        for y in range(2011, 2021):
            ystart = first_day_of_week(y, day_of_week="Wednesday") 
            dates = [ystart + timedelta(days=x) for x in range(0, 364, 7)]
            multiyear_dates += dates

        # Remove duplicates (TODO: question, will there ever be duplicates?)
        # return sorted(set(multiyear_dates), key=lambda x: multiyear_dates.index(x)) # Could improve efficiency 
        return multiyear_dates

    elif date_str == "std_paper_half":
        ''' 
        Paper performance evaluation period from 2011-2020. A year period for 
        year yyyy is defined as 26 predictions biweekly starting from the first Wednesday in January 
        '''
        multiyear_dates = []
        for y in range(2011, 2021):
            ystart = first_day_of_week(y, day_of_week="Wednesday") 
            dates = [ystart + timedelta(days=x) for x in range(0, 364, 14)]
            multiyear_dates += dates

        # Remove duplicates (TODO: question, will there ever be duplicates?)
        # return sorted(set(multiyear_dates), key=lambda x: multiyear_dates.index(x)) # Could improve efficiency 
        return multiyear_dates

    elif date_str == "std_paper_eval":
        ''' 
        Evaluation period for model tuning for paper, daily from 2007-2020. 
        '''
        first_year = 2007
        number_of_days = 365 * 10 + 366 * 4

    elif "-" in date_str:
        # Input is a string of the form '20170101-20180130'
        first_date, last_date = date_str.split("-")
        first_date = string_to_dt(first_date)
        last_date = string_to_dt(last_date)
        dates = [
            first_date + timedelta(days=x)
            for x in range(0, (last_date - first_date).days + 1)
        ]
        return dates
    elif "," in date_str:
        # Input is a string of the form '20170101,20170102,20180309'
        dates = [datetime.strptime(x.strip(), "%Y%m%d") for x in date_str.split(",")]
        return dates
    elif len(date_str) == 6:
        year = int(date_str[0:4])
        month = int(date_str[4:6])

        first_date = datetime(year=year, month=month, day=1)
        if month == 12:
            last_date = datetime(year=year+1, month=1, day=1)
        else:
            last_date = datetime(year=year, month=month+1, day=1)
        dates = [
            first_date + timedelta(days=x)
            for x in range(0, (last_date-first_date).days)
        ]
        return dates
    elif len(date_str) == 8:
        # Input is a string of the form '20170101', representing a single target date
        dates = [datetime.strptime(date_str.strip(), "%Y%m%d")]
        return dates
    else:
        raise NotImplementedError("Date string provided cannot be transformed "
                                  "into list of target dates.")

    # Return standard set of dates
    first_date = datetime(year=first_year, month=1, day=1)
    dates = [first_date + timedelta(days=x) for x in range(number_of_days)]
    return dates


def get_task_metrics_dir(
    model="spatiotemporal_mean", submodel=None, gt_id="contest_tmp2m", horizon="34w",
    target_dates=None
):
    """Returns the directory in which evaluation metrics for a given submodel
    or model are stored

    Args:
       model: string model name
       submodel: string submodel name or None; if None, returns metrics
         directory associated with selected submodel or returns None if no
         submodel selected
       gt_id: contest_tmp2m or contest_precip
       horizon: 34w or 56w
    """
    if submodel is None:
        submodel = get_selected_submodel_name(model=model, gt_id=gt_id, horizon=horizon,
                        target_dates=target_dates)
        if submodel is None:
            return None
    return os.path.join(
        "eval", "metrics", model, "submodel_forecasts", submodel, f"{gt_id}_{horizon}"
    )


def get_contest_task_metrics_dir(
    model="spatiotemporal_mean", gt_id="contest_tmp2m", horizon="34w"
):
    """Returns the directory in which evaluation metrics for a given model contest
    prediction are stored

    Args:
       model: string model name
       gt_id: contest_tmp2m or contest_precip
       horizon: 34w or 56w
    """
    return os.path.join(
        "eval", "metrics", model, "contest_forecasts", f"{gt_id}_{horizon}"
    )


def load_metric(
    model="spatiotemporal_mean",
    submodel=None,
    gt_id="contest_tmp2m",
    horizon="34w",
    target_dates="std_test",
    metric="rmse",
):
    """Loads metric data stored by src/eval/batch_metrics.py;
    returns None if requested data is unavailable.

    Args:
      model - model name
      submodel - if None, loads metric data stored for model (e.g.,
        for the model's selected submodel); otherwise loads metric
        data for requested submodel
      gt_id - "contest_tmp2m" or "contest_precip"
      horizon - "34w" or "56w"
      target_dates - a valid input to get_target_dates
      metric - "rmse", "skill", or "score"
    """
    if submodel is None:
        # Identify the selected submodel name for this model
        submodel = get_selected_submodel_name(model=model, gt_id=gt_id, horizon=horizon, 
            target_dates=target_dates)
    # If there is no selected submodel, return None
    if submodel is None:
        return None
    # Load and return metric data if it exists
    task = f"{gt_id}_{horizon}"
    filename = os.path.join(
        "eval",
        "metrics",
        model,
        "submodel_forecasts",
        submodel,
        task,
        f"{metric}-{task}-{target_dates}.h5",
    )
    if os.path.exists(filename):
        return pd.read_hdf(filename)
    return None

def contest_year(target_date_obj, horizon, dow=1):
    """Returns the contest year associated with a target_date_obj, where
    a contest starts on the last Wednesday in October of a given year for 34w
    two weeks after that for 56w task.

    Args:
      target_date_obj: target date as a datetime object
      horizon: "34w" or "56w" indicating contest forecast horizon
      dow: day of week, 1 = Tuesday, 2 = Wednesday
    """
    year = target_date_obj.year
    contest_start, contest_end = contest_start_end(horizon, year, dow)
    if target_date_obj >= contest_start:
        return year
    else:
        return year-1

def contest_start_end(horizon, year=2019, dow=1):
    """Returns the contest start and end date (inclusive) for a given year and horizon, 
    where a contest starts on the last <dow> in Octobor of a given year for the 34w task
    and two weeks after that for the 56w task.

    Args:
      target_date_obj: target date as a datetime object
      horizon: "34w" or "56w" indicating contest forecast horizon
      dow: day of week, 1 = Tuesday, 2 = Wednesday
    """
    # Find Wednesday in last 7 days of October each year
    for d in range(25, 32):
        date = datetime(year=year, month=10, day=d)
        #if date.weekday() == 1: # Tuesday
        if date.weekday() == dow: 
            contest_start = date
            break

    contest_end = contest_start + timedelta(days=364)
    # Remove first or last two-weeks from contest period for 34w and 56w respectively 
    if horizon == "12w":
        contest_end -= timedelta(weeks=4)
    elif horizon == "34w":
        contest_end -= timedelta(weeks=2)
    elif horizon == "56w":
        contest_start += timedelta(weeks=2)
    else:
        pass
        #printf(f"Unknown horizon {horizon}. Including full period.")
    return contest_start, contest_end

def contest_quarter_start_dates(horizon, year=2019, dow=1):
    """Returns the start day and month for each contest quarter associated with a
    given year as a list of datetime objects

    Args:
      horizon: "34w" or "56w" indicating contest forecast horizon
      year: year to associate with each quarter start day and month
      dow: day of week, 1 = Tuesday, 2 = Wednesday
    """
    # Get contest start/end
    cs, ce = contest_start_end(horizon, year, dow)

    # Set contest offsets
    q0 = 0 # offset in of q0 start from contest_start, in weeks
    q1 = 14 + q0 # offset of q1 from q0 start, in weeks
    q2 = 12 + q1 # offset of q2 from q1 start, in weeks
    q3 = 12 + q2 # offset of q3 from q2 start, in weeks

    # Get quarter offsets and return
    q_duration = [timedelta(weeks=q0), timedelta(weeks=q1), timedelta(weeks=q2), timedelta(weeks=q3)]
    return  [cs + q for q in q_duration]

def contest_quarter(target_date_obj, horizon, dow=1):
    """Returns the frii contest quarter (coded as 0,1,2,3) in which a given
    target datetime object lies.

    Args:
      target_date_obj: target date as a datetime object
      horizon: "34w" or "56w" indicating contest forecast horizon
      dow: day of week, 1 = Tuesday, 2 = Wednesday
    """
    yy = contest_year(target_date_obj, horizon, dow)
    quarter_starts = contest_quarter_start_dates(horizon, yy, dow)
    for ii in range(0, 3):
        if (target_date_obj >= quarter_starts[ii]) and (target_date_obj < quarter_starts[ii + 1]):
            return ii
    # Otherwise, date lies in last quarter
    return 3

def year_quarter(target_date_obj):
    """Returns the yearly quarter (coded as 0,1,2,3) in which a given
    target datetime object lies.

    Args:
      target_date_obj: target date as a datetime object
    """
    m = target_date_obj.month
    if m >= 12 or m <= 2: # December - February
        return 0
    elif m >= 3 and m <= 5: # March - May
        return 1
    elif m >= 6 and m <= 8: # June - August
        return 2
    elif m >= 9 and m <= 11: #  September - November
        return 3
    else:
        raise ValueError(f"Invalid month {m}")
