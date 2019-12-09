BRIDGES = [
    # Manhattan to Bronx
    (42, 247),
    (74, 168),
    (194, 168),
    (120, 119),
    (120, 235),
    (127, 136),
    (153, 220),
    (128, 220),
    # Manhattan to Queens
    (194, 8),
    (140, 193),
    (233, 145),
    (202, 193),
    # Manhattan to Brooklyn
    (232, 256),
    (45, 66),
    (209, 66),
    (12, 195),
    # Brooklyn to Staten Island
    (14, 6),
    # Queens to Bronx
    (252, 208),
    (15, 208),

    # The following are not actual bridges,
    # but instead exist to compensate for the lack
    # of information on NJ
    # Manhattan to EWR
    (125, 1),
    (246, 1),
    (243, 1),
    # EWR to Staten Island
    (23, 1),
    # Manhattan to Staten Island
    (125, 187),
    (246, 187),
    (243, 187),
    ]

SUPER_BRIDGES = {
    1: {
        2: [
            # Manhattan to Queens
            (194, 8),
            (140, 193),
            (233, 145),
            (202, 193),
            # Manhattan to Brooklyn
            (232, 256),
            (45, 66),
            (209, 66),
            (12, 195),
            # Bronx to Queens
            (208, 252),
            (208, 15)
        ],
        3: [
            # EWR to Staten Island
            (23, 1),
            # Manhattan to Staten Island
            (125, 187),
            (246, 187),
            (243, 187)
        ]   
    },

    2: {
        1: [
            # Queens to Manhattan
            (8, 194),
            (193, 140),
            (145, 233),
            (193, 202),
            # Brokklyn to Manhattan
            (256, 232), 
            (66, 45),
            (66, 209),
            (195, 12),
            # Queens to Bronx
            (252, 208), 
            (15, 208)
        ],
        3: [
            # Brooklyn to Staten Island
            (14, 6)
        ]
    },

    3: {
        1: [
            # Staten Island to EWR
            (1, 23),
            # Staten Island to Manhattan
            (187, 125),
            (187, 246),
            (187, 243)
        ],
        2: [
            # Staten Island to Brooklyn
            (6, 14)
        ]
    }
}
