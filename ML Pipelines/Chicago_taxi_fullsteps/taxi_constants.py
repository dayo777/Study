# transform Preprocessing function
import tensorflow as tf
import taxi_constants
import tensorflow_transform as tft


tf.constant

_taxi_constants_module_file = 'taxi_constants.py'

# %%writefile {_taxi_constants_module_file}

# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour', 'trip_start_day', 'trip_start_month',
    'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
    'dropoff_community_area'
]

DENSE_FLOAT_FEATURE_KEYS = ['trip_miles', 'fare', 'trip_seconds']

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10

BUCKET_FEATURE_KEYS = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = [
    'payment_type',
    'company',
]

# Keys
LABEL_KEY = 'tips'
FARE_KEY = 'fare'


# this is the PREPROCESSING_FN that takes raw data as input, and return raw inputs as features
_taxi_transform_module_file = 'taxi_transform.py'

# %%writefile {_taxi_transform_module_file}

_DENSE_FLOAT_FEATURE_KEYS = taxi_constants.DENSE_FLOAT_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = taxi_constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = taxi_constants.VOCAB_SIZE
_OOV_SIZE = taxi_constants.OOV_SIZE
_FEATURE_BUCKET_COUNT = taxi_constants.FEATURE_BUCKET_COUNT
_BUCKET_FEATURE_KEYS = taxi_constants.BUCKET_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = taxi_constants.CATEGORICAL_FEATURE_KEYS
_FARE_KEY = taxi_constants.FARE_KEY
_LABEL_KEY = taxi_constants.LABEL_KEY

def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.
    Args:
        inputs: map from feature keys to raw not-yet-transformed features.
    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        # If sparse make it dense, setting nan's to 0 or '', and apply zscore.
        outputs[key] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key])
        )

    for key in _VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        outputs[key] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]), top_k=_VOCAB_SIZE, 
            num_oov_buckets=_OOV_SIZE
        )

    for key in _BUCKET_FEATURE_KEYS:
        outputs[key] = tft.bucketize(_fill_in_missing(inputs[key]), _FEATURE_BUCKET_COUNT)

    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[key] = _fill_in_missing(inputs[key])

    # was this passenger a big tipper?
    taxi_fare = _fill_in_missing(inputs[_FARE_KEY])
    tips = _fill_in_missing(inputs[_LABEL_KEY])
    outputs[_LABEL_KEY] = tf.where(
        tf.math.is_nan(taxi_fare), tf.cast(tf.zeros_like(taxi_fare), tf.int64),
        # test if the tip was > 20% of the fare.
        tf.cast(tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64)
    ) 
    return outputs



def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
    Args:
        x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
        in the second dimension.
    Returns:
        A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value
        ), axis=1
    )
