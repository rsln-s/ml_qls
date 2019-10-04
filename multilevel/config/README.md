# Configuration files

Configuration files for ml_gp in INI format. All parameters correspond to the ones specified in `python ml_gp -h`

Header is optional

# Examples

Can be run with graph specified in INI file:

`python ml_gp.py ../config/mod_via_gp_arenas_jazz.ini`

Command line parameters can be used to augment / override the config file parameters:

`python ml_gp.py -c ../config/gp.ini ../data/GP/uk.graph`

