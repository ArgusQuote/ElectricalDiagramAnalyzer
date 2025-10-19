#-----------------------------------------------------
# Start Disconnects
class Disconnect():
    def __init__(self):
        # Initialize EasyOCR reader
        self.bool = False

        # Disconnect Library Lookup
        self.disconnectLibrary = {
            # GENERAL DUTY FUSIBLE
            ('GENERAL_DUTY', True, 2, 240, 30, 'NEMA1'):  ("D211N", "PK3GTA1", None),
            ('GENERAL_DUTY', True, 2, 240, 30, 'NEMA3R'): ("D211NRB", "PK3GTA1", None),
            ('GENERAL_DUTY', True, 2, 240, 60, 'NEMA1'):  ("D222N", "GTK03", None),
            ('GENERAL_DUTY', True, 2, 240, 60, 'NEMA3R'): ("D222NRB", "GTK03", None),
            ('GENERAL_DUTY', True, 2, 240, 100, 'NEMA1'):  ("D223N", "GTK0610", None),
            ('GENERAL_DUTY', True, 2, 240, 100, 'NEMA3R'): ("D223NRB", "GTK0610", None),
            ('GENERAL_DUTY', True, 2, 240, 200, 'NEMA1'):  ("D224N", "PKOGTA2", None),
            ('GENERAL_DUTY', True, 2, 240, 200, 'NEMA3R'): ("D224NRB", "PKOGTA2", None),
            ('GENERAL_DUTY', True, 2, 240, 400, 'NEMA1'):  ("D225N", "PKOGTA2", None),
            ('GENERAL_DUTY', True, 2, 240, 400, 'NEMA3R'): ("D225NR", "PKOGTA2", None),
            ('GENERAL_DUTY', True, 2, 240, 600, 'NEMA1'):  ("D226N", "PKOGTA3", None),
            ('GENERAL_DUTY', True, 2, 240, 600, 'NEMA3R'): ("D226NR", "PKOGTA3", None),
            ('GENERAL_DUTY', True, 3, 240, 30, 'NEMA1'):  ("D321N", "PK3GTA1", None),
            ('GENERAL_DUTY', True, 3, 240, 30, 'NEMA3R'): ("D321NRB", "PK3GTA1", None),
            ('GENERAL_DUTY', True, 3, 240, 60, 'NEMA1'):  ("D322N", "GTK03", None),
            ('GENERAL_DUTY', True, 3, 240, 60, 'NEMA3R'): ("D322NRB", "GTK03", None),
            ('GENERAL_DUTY', True, 3, 240, 100, 'NEMA1'):  ("D323N", "GTK0610", None),
            ('GENERAL_DUTY', True, 3, 240, 100, 'NEMA3R'): ("D323NRB", "GTK0610", None),
            ('GENERAL_DUTY', True, 3, 240, 200, 'NEMA1'):  ("D324N", "PKOGTA2", None),
            ('GENERAL_DUTY', True, 3, 240, 200, 'NEMA3R'): ("D324NRB", "PKOGTA2", None),
            ('GENERAL_DUTY', True, 3, 240, 400, 'NEMA1'):  ("D325N", "PKOGTA2", None),
            ('GENERAL_DUTY', True, 3, 240, 400, 'NEMA3R'): ("D325NR", "PKOGTA2", None),
            ('GENERAL_DUTY', True, 3, 240, 600, 'NEMA1'):  ("D326N", "PKOGTA3", None),
            ('GENERAL_DUTY', True, 3, 240, 600, 'NEMA3R'): ("D326NR", "PKOGTA3", None),
            # GENERAL DUTY NON-FUSIBLE (pole input ignored — always 3P frames)
            ('GENERAL_DUTY', False, None, 240, 30, 'NEMA1'):  ("DU321", "PK3GTA1", None),
            ('GENERAL_DUTY', False, None, 240, 30, 'NEMA3R'): ("DU321RB", "PK3GTA1", None),
            ('GENERAL_DUTY', False, None, 240, 60, 'NEMA1'):  ("DU322", "GTK03", None),
            ('GENERAL_DUTY', False, None, 240, 60, 'NEMA3R'): ("DU322RB", "GTK03", None),
            ('GENERAL_DUTY', False, None, 240, 100, 'NEMA1'):  ("DU323", "GTK0610", "SN0610"),
            ('GENERAL_DUTY', False, None, 240, 100, 'NEMA3R'): ("DU323RB", "GTK0610", "SN0610"),
            ('GENERAL_DUTY', False, None, 240, 200, 'NEMA1'):  ("DU324", "PKOGTA2", "SN20A"),
            ('GENERAL_DUTY', False, None, 240, 200, 'NEMA3R'): ("DU324RB", "PKOGTA2", "SN20A"),
            ('GENERAL_DUTY', False, None, 240, 400, 'NEMA1'):  ("DU325", "PKOGTA2", None),
            ('GENERAL_DUTY', False, None, 240, 600, 'NEMA1'):  ("DU326", "PKOGTA2", "D600SN"),
            # HEAVY DUTY NON-FUSIBLE
            ("HEAVY_DUTY", False, 2, 600, 400, "NEMA1"): ("HU265", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", False, 2, 600, 600, "NEMA1"): ("HU266", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", False, 2, 600, 800, "NEMA1"): ("HU267", "PKOGTA7", "H800SNE4"),
            ("HEAVY_DUTY", False, 2, 600, 1200, "NEMA1"): ("HU268", "PKOGTA8", "H1200SNE4"),
            ("HEAVY_DUTY", False, 2, 600, 400, "NEMA3R"): ("HU265R", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", False, 2, 600, 600, "NEMA3R"): ("HU266R", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", False, 2, 600, 800, "NEMA3R"): ("HU267R", "PKOGTA7", "H800SNE4"),
            ("HEAVY_DUTY", False, 2, 600, 1200, "NEMA3R"): ("HU268R", "PKOGTA8", "H1200SNE4"),
            ("HEAVY_DUTY", False, 3, 600, 30, "NEMA1"): ("VHU361", "GTK03", "SN03"),
            ("HEAVY_DUTY", False, 3, 600, 60, "NEMA1"): ("VHU362", "GTK0610", "SN0610"),
            ("HEAVY_DUTY", False, 3, 600, 100, "NEMA1"): ("VHU363", "GTK0610", "SN0610"),
            ("HEAVY_DUTY", False, 3, 600, 200, "NEMA1"): ("VHU364", "PKOGTA2", "SN20A"),
            ("HEAVY_DUTY", False, 3, 600, 400, "NEMA1"): ("HU365", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", False, 3, 600, 600, "NEMA1"): ("HU366", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", False, 3, 600, 800, "NEMA1"): ("HU367", "PKOGTA7", "H800SNE4"),
            ("HEAVY_DUTY", False, 3, 600, 1200, "NEMA1"): ("HU368", "PKOGTA8", "H1200SNE4"),
            ("HEAVY_DUTY", False, 3, 600, 30, "NEMA3R"): ("VHU361RB", "GTK03", "SN03"),
            ("HEAVY_DUTY", False, 3, 600, 60, "NEMA3R"): ("VHU362RB", "GTK0610", "SN0610"),
            ("HEAVY_DUTY", False, 3, 600, 100, "NEMA3R"): ("VHU363RB", "GTK0610", "SN0610"),
            ("HEAVY_DUTY", False, 3, 600, 200, "NEMA3R"): ("VHU364R", "PKOGTA2", "SN20A"),
            ("HEAVY_DUTY", False, 3, 600, 400, "NEMA3R"): ("HU365R", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", False, 3, 600, 600, "NEMA3R"): ("HU366R", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", False, 3, 600, 800, "NEMA3R"): ("HU367R", "PKOGTA7", "H800SNE4"),
            ("HEAVY_DUTY", False, 3, 600, 1200, "NEMA3R"): ("HU368R", "PKOGTA8", "H1200SNE4"),
            # HEAVY DUTY FUSIBLE
            ("HEAVY_DUTY", True, 2, 600, 400, "NEMA1"): ("H265", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", True, 2, 600, 600, "NEMA1"): ("H266", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", True, 2, 600, 800, "NEMA1"): ("H267", "PKOGTA7", "H800SNE4"),
            ("HEAVY_DUTY", True, 2, 600, 1200, "NEMA1"): ("H268", "PKOGTA8", "H1200SNE4"),
            ("HEAVY_DUTY", True, 2, 600, 400, "NEMA3R"): ("H265R", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", True, 2, 600, 600, "NEMA3R"): ("H266R", "PKOGTA2", "H600SN"),
            ("HEAVY_DUTY", True, 2, 600, 800, "NEMA3R"): ("H267R", "PKOGTA7", "H800SNE4"),
            ("HEAVY_DUTY", True, 2, 600, 1200, "NEMA3R"): ("H268R", "PKOGTA8", "H1200SNE4"),
            ("HEAVY_DUTY", True, 3, 600, 30, "NEMA1"): ("VH361", "GTK03", None),
            ("HEAVY_DUTY", True, 3, 600, 60, "NEMA1"): ("VH362", "GTK0610", None),
            ("HEAVY_DUTY", True, 3, 600, 100, "NEMA1"): ("VH363", "GTK0610", None),
            ("HEAVY_DUTY", True, 3, 600, 200, "NEMA1"): ("VH364", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 400, "NEMA1"): ("H365", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 600, "NEMA1"): ("H366", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 800, "NEMA1"): ("H367", "PKOGTA7", None),
            ("HEAVY_DUTY", True, 3, 600, 1200, "NEMA1"): ("H368", "PKOGTA8", None),
            ("HEAVY_DUTY", True, 3, 600, 30, "NEMA3R"): ("VH361RB", "GTK03", None),
            ("HEAVY_DUTY", True, 3, 600, 60, "NEMA3R"): ("VH362RB", "GTK0610", None),
            ("HEAVY_DUTY", True, 3, 600, 100, "NEMA3R"): ("VH363RB", "GTK0610", None),
            ("HEAVY_DUTY", True, 3, 600, 200, "NEMA3R"): ("VH364R", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 400, "NEMA3R"): ("H365R", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 600, "NEMA3R"): ("H366R", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 800, "NEMA3R"): ("H367R", "PKOGTA7", None),
            ("HEAVY_DUTY", True, 3, 600, 1200, "NEMA3R"): ("H368R", "PKOGTA8", None),
            ("HEAVY_DUTY", True, 3, 600, 30, "NEMA1"): ("VH361N", "GTK03", None),
            ("HEAVY_DUTY", True, 3, 600, 60, "NEMA1"): ("VH362N", "GTK0610", None),
            ("HEAVY_DUTY", True, 3, 600, 100, "NEMA1"): ("VH363N", "GTK0610", None),
            ("HEAVY_DUTY", True, 3, 600, 200, "NEMA1"): ("VH364N", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 400, "NEMA1"): ("H365N", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 600, "NEMA1"): ("H366N", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 800, "NEMA1"): ("H367N", "PKOGTA7", None),
            ("HEAVY_DUTY", True, 3, 600, 1200, "NEMA1"): ("H368N", "PKOGTA8", None),
            ("HEAVY_DUTY", True, 3, 600, 30, "NEMA3R"): ("VH361NRB", "GTK03", None),
            ("HEAVY_DUTY", True, 3, 600, 60, "NEMA3R"): ("VH362NRB", "GTK0610", None),
            ("HEAVY_DUTY", True, 3, 600, 100, "NEMA3R"): ("VH363NRB", "GTK0610", None),
            ("HEAVY_DUTY", True, 3, 600, 200, "NEMA3R"): ("VH364NR", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 400, "NEMA3R"): ("H365NR", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 600, "NEMA3R"): ("H366NR", "PKOGTA2", None),
            ("HEAVY_DUTY", True, 3, 600, 800, "NEMA3R"): ("H367NR", "PKOGTA7", None),
            ("HEAVY_DUTY", True, 3, 600, 1200, "NEMA3R"): ("H368NR", "PKOGTA8", None),
        }

    # Disconnect part number generation logic
    def generateDisconnectPartNumber(self, attributes):
        switchType = attributes.get("switchType", "").upper()
        fusible = attributes.get("fusible", False)
        poles = attributes.get("poles")
        voltage = attributes.get("voltage")
        amps = attributes.get("amps")
        neutral = attributes.get("neutral")
        enclosure = attributes.get("enclosure", "").upper()
        groundRequired = attributes.get("groundRequired", False)
        solidNeutral = attributes.get("solidNeutral", False)
        fuseAmperage = attributes.get("fuseAmperage")

        # Normalize the pole input for non-fusible general duty (they use 3P frames)
        if switchType == 'GENERAL_DUTY' and not fusible:
            poles = None

        # Lookup key for the Disconnect_Library
        key = (switchType, fusible, poles, voltage, amps, enclosure)
        if key not in self.disconnectLibrary:
            return "Invalid disconnect configuration."

        partNumber, groundKit, neutralKit = self.disconnectLibrary[key]

        # Fuses (optional)
        fusePartNumber = None
        fuseNoteKey = "Fuses"
        if fusible:
            flnrRatings = [20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 90, 100,
                            110, 125, 150, 175, 200, 225, 250, 300, 350, 400,
                            450, 500, 600, 800]
            flsrRatings = flnrRatings  # same range for FLSR

            if voltage == 600:
                prefix = "FLSR"
                validRatings = flsrRatings
            elif voltage in [120, 240]:
                prefix = "FLNR"
                validRatings = flnrRatings
            else:
                return "Invalid voltage for fuse selection."

            if fuseAmperage not in [None, '']:
                try:
                    fuseAmperage = int(fuseAmperage)
                except ValueError:
                    return f"Invalid fuse amperage input: '{fuseAmperage}' is not a number."

                if fuseAmperage not in validRatings:
                    return f"Invalid fuse amperage '{fuseAmperage}' for {voltage}V. Valid options: {validRatings}"
            else:
                fuseAmperage = amps
                fuseNoteKey = "Fused disconnect detected. Recommended fuses based on amperage"

            fusePartNumber = f"{prefix}{str(fuseAmperage).zfill(3)}"

        # Build the output
        result = {
            "Part Number": partNumber
        }

        if groundRequired and groundKit:
            result["Grounding Kit"] = groundKit
        if solidNeutral and neutralKit:
            result["Solid Neutral Kit"] = neutralKit
        if fusePartNumber:
            result[fuseNoteKey] = f"{poles if poles else 3} - {fusePartNumber}"

        return result

# End Disconnects

#--------------------------------------------------------

# Start Breakers

# Breaker Selection Logic for QO, QOB, HOM, QOH, QH, QHB, QO-VH, and QOB-VH breakers
class breakerSelector:
    def __init__(self, attributes):
        self.breakerType = attributes.get("breakerType")
        self.poles = attributes.get("poles")
        self.amperage = attributes.get("amperage")
        self.interruptionRating = attributes.get("interruptionRating")
        self.specialFeatures = attributes.get("specialFeatures", "").upper()
        self.iline = attributes.get("iline", False)
        self.hiddenKRating = None

    def selectBreaker(self):
        if not self.validateBreakerType():
            return {"Error": "Invalid breaker type. Valid options: ['QO', 'QOB', 'HOM', 'QOH', 'QH', 'QHB', 'QO-VH', 'QOB-VH']"}

        if not self.validatePoles():
            return {"Error": f"Invalid poles selection for {self.breakerType} breakers."}

        if not self.validateAmperage():
            validAmperages = sorted(self.getAmperageOptionsByPoles().keys())
            return {"Error": f"Invalid amperage selection. Valid options for {self.poles}P {self.breakerType} breakers: {validAmperages}"}

        if not self.validateInterruptionRating():
            validRatings = sorted(self.getAmperageOptionsByPoles()[self.amperage])
            return {"Error": f"Invalid interruption rating for {self.amperage}A. Valid options: {validRatings}"}

        if self.specialFeatures and not self.validateSpecialFeatures():
            return {"Error": f"Invalid special feature '{self.specialFeatures}' for {self.amperage}A {self.poles}P breaker."}
        
        if self.breakerType in ["QO", "QOB"] and self.interruptionRating == 22:
            self.breakerType += "-VH"

        if self.breakerType != "HOM":
            self.hiddenKRating = self.getHiddenKRating()
        else:
            self.hiddenKRating = "10k"

        return self.generateBreakerPartNumber()

    def validateBreakerType(self):
        validBreakerTypes = ['QO', 'QOB', 'HOM', 'QOH', 'QH', 'QHB', 'QO-VH', 'QOB-VH']
        return self.breakerType in validBreakerTypes

    def validatePoles(self):
        # QOH is only 2-pole
        if self.breakerType == "QOH":
            return self.poles == 2

        # QO / QOB (plug-on or bolt-on) and their VH variants are all 1–3-pole
        if self.breakerType in ["QO", "QOB", "QO-VH", "QOB-VH"]:
            return self.poles in {1, 2, 3}

        # QH / QHB are 1–3-pole
        if self.breakerType in ["QH", "QHB"]:
            return self.poles in {1, 2, 3}

        # HOM is only 1–2-pole
        if self.breakerType == "HOM":
            return self.poles in {1, 2}

        # anything else is invalid
        return False

    def validateAmperage(self):
        amperageMap = self.getAmperageOptionsByPoles()
        return self.amperage in amperageMap

    def validateInterruptionRating(self):
        # For HOM breakers, we ignore any user-supplied value.
        if self.breakerType == "HOM":
            return True
        return True  # For all non-HOM types, the value is ignored and hidden k rating is used.

    def validateSpecialFeatures(self):
        # For QH and QHB, no special features are allowed.
        if self.breakerType in ["QH", "QHB"]:
            return self.specialFeatures == ""
        # For HOM breakers, apply the HOM-specific rules:
        if self.breakerType == "HOM":
            if self.specialFeatures == "":
                return True
            homSpecialFeatures = {
                "CAFI": {1: [15, 20], 2: [15, 20]},
                "PCAFI": {1: [15, 20], 2: [15, 20]},
                "GFI": {1: [15, 20], 2: [15, 20, 25, 30, 35, 40, 45, 50]},
                "EPD": {1: [15, 20], 2: [15, 20, 25, 30, 40, 45, 50]},
                "DF": {1: [15, 20], 2: [15, 20]},
                "PDF": {1: [15, 20], 2: [15, 20]},
                "HM": {1: [15, 20]}
            }
            if self.specialFeatures in homSpecialFeatures:
                allowed = homSpecialFeatures[self.specialFeatures].get(self.poles, [])
                return self.amperage in allowed
            else:
                return False
        # For all other breaker types, use the standard special feature map.
        specialFeatureMap = self.getSpecialFeatureMap()
        return self.specialFeatures in specialFeatureMap.get(self.amperage, [])

    def getAmperageOptionsByPoles(self):
        # QOH mapping (unchanged)
        if self.breakerType == "QOH":
            return { 40: [42], 45: [42], 50: [42], 60: [42], 70: [42], 80: [42], 90: [42], 100: [42], 110: [42], 125: [42],}
        # QH and QHB mapping
        elif self.breakerType in ["QH", "QHB"]:
            return { 15: [65], 20: [65], 25: [65], 30: [65],}
        # QO-VH and QOB-VH mapping
        elif self.breakerType in ["QO-VH", "QOB-VH"]:
            if self.poles == 1:
                return { 15: [22], 20: [22], 25: [22], 30: [22], 40: [22], 50: [22], 60: [22], 70: [22],}
            elif self.poles == 2:
                return { 15: [22], 20: [22], 25: [22], 30: [22], 40: [22], 50: [22], 60: [22], 70: [22], 80: [22], 90: [22], 100: [22], 110: [22], 125: [22], 150: [22], 175: [22], 200: [22],}
            elif self.poles == 3:
                return { 15: [22], 20: [22], 25: [22], 30: [22], 40: [22], 50: [22], 60: [22], 70: [22], 80: [22], 90: [22], 100: [22],}
        # QO and QOB mapping (updated)
        elif self.breakerType in ["QO", "QOB"]:
            if self.poles == 1:
                return { 10: [10], 15: [10], 20: [10], 25: [10], 30: [10], 35: [10], 40: [10], 45: [10], 50: [10], 60: [10], 70: [10],}
            elif self.poles == 2:
                return { 10: [10], 15: [10], 20: [10], 25: [10], 30: [10], 35: [10], 40: [10], 45: [10], 50: [10], 60: [10], 70: [10], 80: [10], 90: [10], 100: [10], 110: [10], 125: [10], 150: [10], 175: [10], 200: [10],}
            elif self.poles == 3:
                return { 10: [10], 15: [10], 20: [10], 25: [10], 30: [10], 35: [10], 40: [10], 45: [10], 50: [10], 60: [10], 70: [10], 80: [10], 90: [10], 100: [10],}
        # HOM mapping (new)
        elif self.breakerType == "HOM":
            if self.poles == 1:
                return { 15: [10], 20: [10], 25: [10], 30: [10], 40: [10], 50: [10],}
            elif self.poles == 2:
                return { 15: [10], 20: [10], 25: [10], 30: [10], 35: [10], 40: [10], 45: [10], 50: [10], 60: [10], 70: [10], 80: [10], 90: [10], 100: [10], 110: [10], 125: [10],}
        else:
            # Fallback (should not occur)
            return {}

    def getSpecialFeatureMap(self):
        # Standard mapping used for non-HOM types.
        return {
            15: ['CAFI', 'PCAFI', 'GFI', 'DF', 'PDF', 'EPD', 'HM'],
            20: ['CAFI', 'PCAFI', 'GFI', 'DF', 'PDF', 'EPD', 'HM'],
            25: ['GFI', 'EPD'],
            30: ['GFI', 'EPD'],
            40: ['GFI', 'EPD'],
            50: ['GFI', 'EPD'],
            60: ['GFI', 'EPD'],
        }

    def getHiddenKRating(self):
        kRatingMap = {
            "QO": "10k",
            "QOB": "10k",
            "QOH": "42k",
            "QO-VH": "22k",
            "QOB-VH": "22k",
            "QH": "65k",
            "QHB": "65k",
            "HOM": "10k"
        }
        return kRatingMap.get(self.breakerType)

    def generateBreakerPartNumber(self):
        output = {}

        if self.breakerType == "QOH":
            qohMapping = {
                40: "QOH240", 45: "QOH245", 50: "QOH250", 60: "QOH260",
                70: "QOH270", 80: "QOH280", 90: "QOH290", 100: "QOH2100",
                110: "QOH2110", 125: "QOH2125",
            }
            partNumber = qohMapping.get(self.amperage)
            if not partNumber:
                return {"Error": f"Invalid amperage for QOH breakers. Valid options: {list(qohMapping.keys())}"}
            output["Part Number"] = partNumber
            output["Interrupting Rating"] = self.hiddenKRating
            return output

        elif self.breakerType in ["QH", "QHB"]:
            mapping = {
                1: {15: f"{self.breakerType}115", 20: f"{self.breakerType}120", 25: f"{self.breakerType}125", 30: f"{self.breakerType}130"},
                2: {15: f"{self.breakerType}215", 20: f"{self.breakerType}220", 25: f"{self.breakerType}225", 30: f"{self.breakerType}230"},
                3: {15: f"{self.breakerType}315", 20: f"{self.breakerType}320", 25: f"{self.breakerType}325", 30: f"{self.breakerType}330"},
            }
            partNumber = mapping.get(self.poles, {}).get(self.amperage)
            if not partNumber:
                valid = list(mapping.get(self.poles, {}).keys())
                return {"Error": f"Invalid amperage for {self.breakerType} breakers. Valid options: {valid}"}
            output["Part Number"] = partNumber
            output["Interrupting Rating"] = self.hiddenKRating
            return output

        else:
            baseType = self.breakerType.replace("-VH", "")
            suffix = "VH" if "-VH" in self.breakerType else ""
            partNumber = f"{baseType}{self.poles}{self.amperage}{suffix}"
            if self.specialFeatures:
                partNumber += self.specialFeatures

            output["Part Number"] = partNumber
            output["Interrupting Rating"] = self.hiddenKRating

            if self.iline and baseType in ['QO', 'QOB']:
                output["HQO Accessory"] = "HQO306 (Takes 4.5 spaces, supports 6 QO breakers)"

            return output

# End Breakers ^

#--------------------------------------------------

# Start E-Frame Breakers
class eFrameBreaker():
    def __init__(self):
        # Initialize EasyOCR reader
        self.bool = False

    # Function to generate NF E‐Frame Breakers
    def generateEBreakerPartNumber(self, attributes):
        """
        Generate the E-Frame breaker part number for NF panelboards.
        Parameters:
        poles: str - one of "1p", "2p", "3p", or "edp" 
                (for edp, these are equipment protection devices and are typically 1p at 277V)
        amperage: int - the amperage rating (in Amps)
        interrupting_rating: int - the interrupting rating in kA; allowed values: 18, 35, or 65
        Returns:
        A string containing the part number or an error message if the inputs are invalid.
        """
        poles = attributes.get("poles")
        amperage = attributes.get("amperage")
        interruptingRating = attributes.get("interruptingRating")

        # Allowed amperages by poles type
        allowedAmperages = {
            "1": [15, 20, 25, 30, 35, 40, 45, 50, 60, 70],
            "2": [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 125],
            "3": [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 125],
            "EDP": [15, 20, 30, 40, 50]
        }
        if poles not in allowedAmperages:
            return {"Error": "Invalid poles type for NF E-Frame breaker. Valid options: 1, 2, 3, EDP"}
        if amperage not in allowedAmperages[poles]:
            return {"Error": f"Invalid amperage {amperage}A for {poles} NF E-Frame breaker."}
        if interruptingRating not in [18, 35, 65]:
            return {"Error": "Invalid interrupting rating; allowed values: 18, 35, 65"}

        # Determine base number from poles:
        # For 1p and edp, use '14'; for 2p use '24'; for 3p use '34'
        if poles in ["1", "EDP"]:
            base = "14"
        elif poles == "2":
            base = "24"
        elif poles == "3":
            base = "34"

        if interruptingRating == 18:
            prefix = "EDB"
        elif interruptingRating == 35:
            prefix = "EGB"
        elif interruptingRating == 65:
            prefix = "EJB"
        
        # For EDP (ground fault protection), append "EPD" at the end
        partNumber = f"{prefix}{base}{amperage:03d}"
        if poles == "EDP":
            partNumber += "EPD"
        
        return {
            "Part Number": partNumber,
            "Interrupting Rating": f"{interruptingRating}kA"
        }

# End E-Frame Breakers ^

#---------------------------------------------------

# Start Loadcenters
class loadcenter():
    def __init__(self):
        # Initialize EasyOCR reader
        self.bool = False

    # Helper function: Generate the additional ground bar kit for HOM load centers
    def generateHomGroundBar(self, homPartNumber, groundBarRequired):
        """
        Given a HOM load center part number (e.g., 'HOM24L70F', 'HOM612L100S', etc.)
        and a flag indicating that a ground bar is required,
        this function returns the additional ground bar kit part number.
        
        If the load center already includes a 'G' (factory-installed ground bar)
        or if no ground bar is required, an appropriate message is returned.
        """
        if not groundBarRequired:
            return {"Message": "No additional ground bar kit required."}
        
        if "G" in homPartNumber:
            return {"Message": "Factory ground bar included (no additional kit required)."}

        mapping = {
            ("HOM24L70F", "HOM24L70S"): "PK3GTA1",
            ("HOM612L100F", "HOM612L100S"): "PK7GTA",
            ("HOM816L125PC",): "PK9GTA",
            ("HOM1224L125PC", "HOM1632L125PC"): "PK15GTA",
            ("HOM2040L125PC",): "PK18GTA",
            ("HOM3060L125PC",): "PK23GTA",
            ("HOM3060L225PC",): "PK23GTA",
            ("HOM4080L225PC", "HOM4284L225PC", "HOM60120L225PC"): "PK27GTA",
            ("HOM816M100PC",): "PK9GTA",
            ("HOM1224M100PC",): "PK15GTA",
            ("HOM2040M100PC",): "PK18GTA",
            ("HOM2448M100PC", "HOM3060M100PC", "HOM2448M125PC", "HOM3060M125PC"): "PK23GTA",
            ("HOM3060M150PC",): "PK23GTA",
            ("HOM2040M200PC",): "PK18GTA",
            ("HOM3060M200PC",): "PK23GTA",
            ("HOM4080M200PC", "HOM4284M200PC", "HOM60120M200PC", "HOM4284L225PC"): "PK27GTA",
            ("HOM3060M200PQC",): "PK23GTA",
            ("HOM4080M200PQC", "HOM4284M200PQC"): "PK27GTA",
            ("HOM24L70RB",): "PK4GTA",
            ("HOM612L100RB",): "PK7GTA",
            ("HOM816L125PRB",): "PK9GTA",
            ("HOM1224M100PRB",): "PK15GTA",
            ("HOM2040L125PRB",): "PK18GTA",
            ("HOM2448M125PRB",): "PK23GTA",
            ("HOM12L225PRB",): "PK9GTA",
            ("HOM1632L225PRB",): "PK15GTA",
            ("HOM2040L225PRB",): "PK18GTA",
            ("HOM3060L225PRB",): "PK23GTA",
            ("HOM4080L225PRB", "HOM4284L225PRB"): "PK27GTA",
            ("HOM816M100PRB",): "PK9GTA",
            ("HOM1224M100PRB",): "PK15GTA",
            ("HOM2040M100PRB",): "PK18GTA",
            ("HOM816M125PRB",): "PK9GTA",
            ("HOM2448M125PRB",): "PK23GTA",
            ("HOM3060M150PRB",): "PK23GTA",
            ("HOM12M200PRB",): "PK9GTA",
            ("HOM2040M200PRB",): "PK18GTA",
            ("HOM3060M200PRB",): "PK23GTA",
            ("HOM4080M200PRB",): "PK27GTA",
            ("HOM816M150PFTRB", "HOM816M200PFTRB"): "PK15GTA",
            ("HOM2040M100PCVP",): "PK18GTA",
            ("HOM2448M100PCVP", "HOM3060M150PCVP"): "PK23GTA",
            ("HOM2040M200PCVP",): "PK18GTA",
            ("HOM3060M200PCVP",): "PK23GTA",
            ("HOM4080M200PCVP",): "PK27GTA",
            ("HOM2040M100PQCVP",): "PK18GTA",
            ("HOM3060M200PQCVP",): "PK23GTA",
            ("HOM4080M200PQCVP",): "PK27GTA",
            ("HOM1224M125PRBVP", "HOM3060M200PRBVP"): "PK23GTA",
        }
        for keyGroup, kit in mapping.items():
            if homPartNumber in keyGroup:
                return {"Ground Bar Kit": kit}
        return {"Error": "Ground bar kit mapping not found for this configuration."}

    # QO Ground Bar Helper Function
    def generateQoGroundBar(self, qoBasePartNumber, groundBarRequired):
        """
        Given a QO base part number (for example, 'QO112M100P') and a flag indicating that
        a ground bar is required, return the corresponding ground bar kit part number.
        
        If the part number already includes a 'G', assume the ground bar is factory-installed.
        """
        if not groundBarRequired:
            return {"Message": "No additional ground bar kit required."}
        if "G" in qoBasePartNumber:
            return {"Message": "Factory ground bar included (no additional kit required)."}
        
        # Mapping from QO base part numbers to ground bar kit part numbers.
        qoGroundBarMapping = {
            'QO112M100P': 'PK9GTA',
            'QO116M100P': 'PK9GTA',
            'QO120M100P': 'PK9GTA',
            'QO124M100P': 'PK15GTA',
            'QO132M100P': 'PK15GTA',
            'QO124M125P': 'PK15GTA',
            'QO132M125P': 'PK15GTA',
            'QO120M150P': 'PK15GTA',
            'QO124M150P': 'PK15GTA',
            'QO130M150P': 'PK15GTA',
            'QO132M150P': 'PK15GTA',
            'QO120M200P': 'PK15GTA',
            'QO124M200P': 'PK15GTA',
            'QO130M200P': 'PK15GTA',
            'QO140M200P': 'PK23GTA',
            'QO142M200P': 'PK18GTA',
            'QO154M200P': 'PK23GTA',
            'QO160M200PC': 'PK27GTA',
            'QO140M225P': 'PK23GTA',
            'QO142M225P': 'PK18GTA',
            'QO130M200PQ': 'PK23GTA',
            'QO142M200PQ': 'PK23GTA',
            'QO154M200PQ': 'PK23GTA',
            'QO112M100PC': 'PK9GTA',
            'QO116M100PC': 'PK9GTA',
            'QO120M100PC': 'PK9GTA',
            'QO124M100PC': 'PK15GTA',
            'QO130M150PC': 'PK15GTA',
            'QO142M150PC': 'PK18GTA',
            'QO130M200PC': 'PK15GTA',
            'QO140M200PC': 'PK23GTA',
            'QO142M200PC': 'PK18GTA',
            'QO154M200PC': 'PK23GTA',
            'QO112M100PRB': 'PK9GTA',
            'QO116M100PRB': 'PK9GTA',
            'QO120M100PRB': 'PK9GTA',
            'QO124M100PRB': 'PK15GTA',
            'QO124M125PRB': 'PK15GTA',
            'QO120M150PRB': 'PK15GTA',
            'QO130M150PRB': 'PK15GTA',
            'QO120M200PRB': 'PK15GTA',
            'QO130M200PRB': 'PK15GTA',
            'QO140M200PRB': 'PK23GTA',
            'QO142M200PRB': 'PK18GTA',
            'QO142M225PRB': 'PK18GTA',
            'QO327M100':    'PK15GTA',
            'QO330MQ125':   'PK18GTA',
            'QO330MQ150':   'PK18GTA',
            'QO342MQ150':   'PK23GTA',
            'QO330MQ200':   'PK18GTA',
            'QO342MQ200':   'PK23GTA',
            'QO342MQ225':   'PK23GTA',
            'QO327M100RB':  'PK15GTA',
            'QO330MQ125RB': 'PK18GTA',
            'QO330MQ150RB': 'PK18GTA',
            'QO330MQ200RB': 'PK18GTA',
            'QO342MQ200RB': 'PK23GTA',
            'QO342MQ225RB': 'PK23GTA',
        }
        if qoBasePartNumber in qoGroundBarMapping:
            return {"Ground Bar Kit": qoGroundBarMapping[qoBasePartNumber]}
        return {"Error": "Ground bar kit mapping not found for QO"}

    # Function to generate Load Center part number
    def generateLoadcenterPartNumber(self, attributes):
        loadCenterType = attributes.get("loadCenterType")
        enclosure = attributes.get("enclosure")
        phasing = attributes.get("phasing")
        typeOfMain = attributes.get("typeOfMain")
        mainsRating = attributes.get("mainsRating")
        poleSpaces = attributes.get("poleSpaces")
        plugOnNeutral = attributes.get("plugOnNeutral")
        coverStyle = attributes.get("coverStyle")
        valuePack = attributes.get("valuePack")
        groundBar = attributes.get("groundBar")
        specialApplication = attributes.get("specialApplication")
        quikGrip = attributes.get("quikGrip")
        busMaterial = attributes.get("busMaterial", "Aluminum")

        # Combined mappings for part number components
        mappings = {
            'typeMap': {'Homeline': 'HOM', 'QO': 'QO'},
            'mainsTypeMap': {'MAIN BREAKER': 'M', 'MAIN LUGS': 'L'},
            'mainsRatingMap': {70: '70', 100: '100', 125: '125', 150: '150', 200: '200', 225: '225'},
            'poleSpacesMap': {4: '24', 8: '48', 12: '612', 16: '816', 24: '1224', 32: '1632', 40: '2040', 48: '2448', 60: '3060', 80: '4080', 84: '4284', 120: '60120'},
            'plugOnNeutralMap': {True: 'P', False: ''},
            'suffixMap': {'Flush': 'F', 'Included': 'C', 'Surface': 'S', 'Combination': 'C', 'Value Pack': 'VP', 'Arc Fault Value Pack': 'A', 'Ground Bar': 'G', 'Feed-Thru Lugs': 'FT', 'Quik-Grip': 'Q'},
            'busMaterialMap': {'Aluminum': '', 'Copper': 'CU'}
        }

        # HOM Section - Generate Box and Interior Part Number
        if loadCenterType in ['Homeline', 'HOM']:
            # List of strictly allowed configurations for HOM (Allowed keys as 11-tuples)
            allowedConfigurationsHom = {
            # Main Lugs, NEMA1, without PON
            ('HOM', 'NEMA1', '1PHASE', 'L', 70, 4, True, 'Flush', False, 'Standard', False): 'HOM24L70F',
            ('HOM', 'NEMA1', '1PHASE', 'L', 70, 4, False, 'Surface', False, 'Standard', False): 'HOM24L70S',
            ('HOM', 'NEMA1', '1PHASE', 'L', 100, 12, False, 'Flush', False, 'Standard', False): 'HOM612L100F',
            ('HOM', 'NEMA1', '1PHASE', 'L', 100, 12, False, 'Surface', False, 'Standard', False): 'HOM612L100S',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 8, False, 'Included', False, 'Standard', False): 'HOM48L125GC',
            # Main Lugs, PON, NEMA1
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 16, True, 'Included', False, 'Standard', False): 'HOM816L125PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 24, True, 'Included', False, 'Standard', False): 'HOM1224L125PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 32, True, 'Included', False, 'Standard', False): 'HOM1632L125PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 40, True, 'Included', False, 'Standard', False): 'HOM2040L125PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 60, True, 'Included', False, 'Standard', False): 'HOM3060L125PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 60, True, 'Included', False, 'Standard', False): 'HOM3060L225PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 80, True, 'Included', False, 'Standard', False): 'HOM4080L225PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 84, True, 'Included', False, 'Standard', False): 'HOM4284L225PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 120, True, 'Included', False, 'Standard', False): 'HOM60120L225PC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 16, True, 'Included', False, 'Standard', False): 'HOM816L125PGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 24, True, 'Included', False, 'Standard', False): 'HOM1224L125PGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 40, True, 'Included', False, 'Standard', False): 'HOM2040L125PGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 48, True, 'Included', False, 'Standard', False): 'HOM2448L125PGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 60, True, 'Included', False, 'Standard', False): 'HOM3060L225PGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 32, True, 'Included', False, 'Standard', False): 'HOM1632L225PGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 40, True, 'Included', False, 'Standard', False): 'HOM2040L225PGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 80, True, 'Included', False, 'Standard', False): 'HOM4080L225PGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 84, True, 'Included', False, 'Standard', False): 'HOM4284L225PGC',
            # Main Breaker, PON, NEMA1
            ('HOM', 'NEMA1', '1PHASE', 'M', 100, 16, True, 'Included', False, 'Standard', False): 'HOM816M100PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 100, 24, True, 'Included', False, 'Standard', False): 'HOM1224M100PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 100, 40, True, 'Included', False, 'Standard', False): 'HOM2040M100PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 100, 48, True, 'Included', False, 'Standard', False): 'HOM2448M100PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 100, 60, True, 'Included', False, 'Standard', False): 'HOM3060M100PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 125, 48, True, 'Included', False, 'Standard', False): 'HOM2448M125PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 125, 60, True, 'Included', False, 'Standard', False): 'HOM3060M125PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 150, 60, True, 'Included', False, 'Standard', False): 'HOM3060M150PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 40, True, 'Included', False, 'Standard', False): 'HOM2040M200PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 60, True, 'Included', False, 'Standard', False): 'HOM3060M200PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 80, True, 'Included', False, 'Standard', False): 'HOM4080M200PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 84, True, 'Included', False, 'Standard', False): 'HOM4284M200PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 120, True, 'Included', False, 'Standard', False): 'HOM60120M200PC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 225, 84, True, 'Included', False, 'Standard', False): 'HOM4284M225PC',
            # PON, Quick Grip, NEMA1
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 48, True, 'Included', False, 'Standard', False): 'HOM2448L125PQGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 60, True, 'Included', False, 'Standard', False): 'HOM3060L125PQGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 60, True, 'Included', False, 'Standard', False): 'HOM3060L225PQGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 80, True, 'Included', False, 'Standard', False): 'HOM4080L225PQGC',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 84, True, 'Included', False, 'Standard', False): 'HOM4284L225PQGC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 60, True, 'Included', False, 'Standard', False): 'HOM3060M200PQC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 80, True, 'Included', False, 'Standard', False): 'HOM4080M200PQC',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 84, True, 'Included', False, 'Standard', False): 'HOM4284M200PQC',
            # Main Lug, NEMA3R
            ('HOM', 'NEMA3R', '1PHASE', 'L', 70, 4, False, 'Included', False, 'Standard', False): 'HOM24L70RB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 100, 12, False, 'Included', False, 'Standard', False): 'HOM612L100RB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 125, 8, False, 'Included', True, 'Standard', False): 'HOM48L125GRB',
            # Main Lug, PON, NEMA3R
            ('HOM', 'NEMA3R', '1PHASE', 'L', 125, 16, True, 'Included', False, 'Standard', False): 'HOM816L125PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 125, 24, True, 'Included', False, 'Standard', False): 'HOM1224L125PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 125, 40, True, 'Included', False, 'Standard', False): 'HOM2040L125PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 125, 48, True, 'Included', False, 'Standard', False): 'HOM2448L125PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 225, 12, True, 'Included', False, 'Standard', False): 'HOM12L225PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 225, 32, True, 'Included', False, 'Standard', False): 'HOM1632L225PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 225, 40, True, 'Included', False, 'Standard', False): 'HOM2040L225PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 225, 60, True, 'Included', False, 'Standard', False): 'HOM3060L225PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 225, 80, True, 'Included', False, 'Standard', False): 'HOM4080L225PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'L', 225, 84, True, 'Included', False, 'Standard', False): 'HOM4284L225PRB',
            # Main Breaker, PON, NEMA3R
            ('HOM', 'NEMA3R', '1PHASE', 'M', 100, 16, True, 'Included', False, 'Standard', False): 'HOM816M100PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 100, 24, True, 'Included', False, 'Standard', False): 'HOM1224M100PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 100, 40, True, 'Included', False, 'Standard', False): 'HOM2040M100PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 125, 16, True, 'Included', False, 'Standard', False): 'HOM816M125PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 125, 48, True, 'Included', False, 'Standard', False): 'HOM2448M125PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 150, 60, True, 'Included', False, 'Standard', False): 'HOM3060M150PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 200, 12, True, 'Included', False, 'Standard', False): 'HOM12M200PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 200, 40, True, 'Included', False, 'Standard', False): 'HOM2040M200PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 200, 60, True, 'Included', False, 'Standard', False): 'HOM3060M200PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 200, 80, True, 'Included', False, 'Standard', False): 'HOM4080M200PRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 150, 16, True, 'Included', False, 'Feed-thru lugs', False): 'HOM816M150PFTRB',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 200, 16, True, 'Included', False, 'Feed-thru lugs', False): 'HOM816M200PFTRB',
            # PON, Value Pack, NEMA1
            ('HOM', 'NEMA1', '1PHASE', 'L', 125, 24, True, 'Included', True, 'Value Pack', False): 'HOM1224L125PGCVP',
            ('HOM', 'NEMA1', '1PHASE', 'L', 225, 60, True, 'Included', True, 'Value Pack', False): 'HOM3060L225PGCVP',
            ('HOM', 'NEMA1', '1PHASE', 'M', 100, 40, True, 'Included', False, 'Value Pack', False): 'HOM2040M100PCVP',
            ('HOM', 'NEMA1', '1PHASE', 'M', 100, 48, True, 'Included', False, 'Value Pack', False): 'HOM2448M100PCVP',
            ('HOM', 'NEMA1', '1PHASE', 'M', 150, 30, True, 'Included', False, 'Value Pack', False): 'HOM3030M150PCVP',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 40, True, 'Included', False, 'Value Pack', False): 'HOM2040M200PCVP',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 60, True, 'Included', False, 'Value Pack', False): 'HOM3060M200PCVP',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 80, True, 'Included', False, 'Value Pack', False): 'HOM4080M200PCVP',
            # PON, Quick Grip, Value Pack, NEMA1
            ('HOM', 'NEMA1', '1PHASE', 'M', 100, 40, True, 'Included', False, 'Value Pack', False): 'HOM2040M100PQCVP',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 60, True, 'Included', False, 'Value Pack', False): 'HOM3060M200PQCVP',
            ('HOM', 'NEMA1', '1PHASE', 'M', 200, 80, True, 'Included', False, 'Value Pack', False): 'HOM4080M200PQCVP',
            # PON, Value Pack, NEMA3R, Main Breaker
            ('HOM', 'NEMA3R', '1PHASE', 'M', 125, 24, True, 'Included', False, 'Value Pack', False): 'HOM1224M125PRBVP',
            ('HOM', 'NEMA3R', '1PHASE', 'M', 200, 60, True, 'Included', False, 'Value Pack', False): 'HOM3060M200PRBVP',
        }

        # For HOM configurations, remove ground_bar from the key.
            configKey = (loadCenterType, enclosure, phasing, typeOfMain, mainsRating, poleSpaces,
                        plugOnNeutral, coverStyle, valuePack, specialApplication, quikGrip)
            if configKey in allowedConfigurationsHom:
                homPartNumber = allowedConfigurationsHom[configKey]
                result = {"Part Number": homPartNumber}
                if groundBar is True:
                    gbResult = self.generateHomGroundBar(homPartNumber, True)
                    result.update(gbResult)
                return result
            else:
                return "Invalid configuration for HOM Load Center"

        # QO Section - Generate Box and Interior Part Number
        elif loadCenterType == 'QO':
            allowedConfigurationsQo = {
                # Main Lugs, PON, NEMA1, Copper, 200A
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 12, True, 'Flush', False, 'Standard', False): ('QO112L200PG', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 12, True, 'Surface', False, 'Standard', False): ('QO112L200PG', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 12, True, 'Mono-Flat', False, 'Standard', False): ('QO112L200PG', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 24, True, 'Flush', False, 'Standard', False): ('QO124L200PG', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 24, True, 'Surface', False, 'Standard', False): ('QO124L200PG', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 24, True, 'Mono-Flat', False, 'Standard', False): ('QO124L200PG', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 30, True, 'Flush', False, 'Standard', False): ('QO130L200PG', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 30, True, 'Surface', False, 'Standard', False): ('QO130L200PG', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 30, True, 'Mono-Flat', False, 'Standard', False): ('QO130L200PG', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 40, True, 'Flush', False, 'Standard', False): ('QO140L200PG', 'QOC40UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 40, True, 'Surface', False, 'Standard', False): ('QO140L200PG', 'QOC40US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 40, True, 'Mono-Flat', False, 'Standard', False): ('QO140L200PG', 'QOCMF40UCW'),
                # Main Lugs, PON, NEMA1, Copper, 225A
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 42, True, 'Flush', False, 'Standard', False): ('QO142L225PG', 'QOC42UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 42, True, 'Surface', False, 'Standard', False): ('QO142L225PG', 'QOC42US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 42, True, 'Mono-Flat', False, 'Standard', False): ('QO142L225PG', 'QOCMF42UCW'),
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 54, True, 'Flush', False, 'Standard', False): ('QO154L225PG', 'QOC54UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 54, True, 'Surface', False, 'Standard', False): ('QO154L225PG', 'QOC54US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 54, True, 'Mono-Flat', False, 'Standard', False): ('QO154L225PG', 'QOCMF54UCW'),
                # Main Breaker, PON, NEMA1, Copper, 100A
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 12, True, 'Flush', False, 'Standard', False): ('QO112M100P', 'QOC12UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 12, True, 'Surface', False, 'Standard', False): ('QO112M100P', 'QOC12US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 16, True, 'Flush', False, 'Standard', False): ('QO116M100P', 'QOC20U100F'),
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 16, True, 'Surface', False, 'Standard', False): ('QO116M100P', 'QOC20U100S'),
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 20, True, 'Flush', False, 'Standard', False): ('QO120M100P', 'QOC20U100F'),
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 20, True, 'Surface', False, 'Standard', False): ('QO120M100P', 'QOC20U100S'),
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 24, True, 'Flush', False, 'Standard', False): ('QO124M100P', 'QOC24UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 24, True, 'Surface', False, 'Standard', False): ('QO124M100P', 'QOC24US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 32, True, 'Flush', False, 'Standard', False): ('QO132M100P', 'QOC32UF'),
                # Main Breaker, PON, NEMA1, Copper, 125A
                ('QO', 'NEMA1', '1PHASE', 'M', 125, 24, True, 'Flush', False, 'Standard', False): ('QO124M125P', 'QOC24UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 125, 24, True, 'Surface', False, 'Standard', False): ('QO124M125P', 'QOC24US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 125, 32, True, 'Flush', False, 'Standard', False): ('QO132M125P', 'QOC32UF'),
                # Main Breaker, PON, NEMA1, Copper, 150A
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 20, True, 'Flush', False, 'Standard', False): ('QO120M150P', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 20, True, 'Surface', False, 'Standard', False): ('QO120M150P', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 20, True, 'Mono-Flat', False, 'Standard', False): ('QO120M150P', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 24, True, 'Flush', False, 'Standard', False): ('QO124M150P', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 24, True, 'Surface', False, 'Standard', False): ('QO124M150P', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 24, True, 'Mono-Flat', False, 'Standard', False): ('QO124M150P', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 30, True, 'Flush', False, 'Standard', False): ('QO130M150P', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 30, True, 'Surface', False, 'Standard', False): ('QO130M150P', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 30, True, 'Mono-Flat', False, 'Standard', False): ('QO130M150P', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 32, True, 'Flush', False, 'Standard', False): ('QO132M150P', 'QOC40UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 32, True, 'Surface', False, 'Standard', False): ('QO132M150P', 'QOC40US'),
                # Main Breaker, PON, NEMA1, Copper, 20A
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 20, True, 'Flush', False, 'Standard', False): ('QO120M200P', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 20, True, 'Surface', False, 'Standard', False): ('QO120M200P', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 20, True, 'Mono-Flat', False, 'Standard', False): ('QO120M200P', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 24, True, 'Flush', False, 'Standard', False): ('QO124M200P', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 24, True, 'Surface', False, 'Standard', False): ('QO124M200P', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 24, True, 'Mono-Flat', False, 'Standard', False): ('QO124M200P', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 30, True, 'Flush', False, 'Standard', False): ('QO130M200P', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 30, True, 'Surface', False, 'Standard', False): ('QO130M200P', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 30, True, 'Mono-Flat', False, 'Standard', False): ('QO130M200P', 'QOCMF30UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 40, True, 'Flush', False, 'Standard', False): ('QO140M200P', 'QOC40UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 40, True, 'Surface', False, 'Standard', False): ('QO140M200P', 'QOC40US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 42, True, 'Flush', False, 'Standard', False): ('QO142M200P', 'QOC42UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 42, True, 'Surface', False, 'Standard', False): ('QO142M200P', 'QOC42US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 42, True, 'Mono-Flat', False, 'Standard', False): ('QO142M200P', 'QOCMF42UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 54, True, 'Flush', False, 'Standard', False): ('QO154M200P', 'QOC54UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 54, True, 'Mono-Flat', False, 'Standard', False): ('QO154M200P', 'QOCMF54UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 60, True, 'Combination', False, 'Standard', False): ('QO160M200PC', ''),
                # Main Breaker, PON, NEMA1, Copper, 225A
                ('QO', 'NEMA1', '1PHASE', 'M', 225, 40, True, 'Flush', False, 'Standard', False): ('QO140M225P', 'QOC42UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 225, 40, True, 'Surface', False, 'Standard', False): ('QO140M225P', 'QOC42US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 225, 40, True, 'Mono-Flat', False, 'Standard', False): ('QO140M225P', 'QOCMF42UCW'),
                ('QO', 'NEMA1', '1PHASE', 'M', 225, 42, True, 'Flush', False, 'Standard', False): ('QO142M225P', 'QOC42UF'),
                ('QO', 'NEMA1', '1PHASE', 'M', 225, 42, True, 'Surface', False, 'Standard', False): ('QO142M225P', 'QOC42US'),
                ('QO', 'NEMA1', '1PHASE', 'M', 225, 42, True, 'Mono-Flat', False, 'Standard', False): ('QO142M225P', 'QOCMF42UCW'),
                # PON, Quick Grip, NEMA1, Copper
                # Main Lugs, 125A
                ('QO', 'NEMA1', '1PHASE', 'L', 125, 24, True, 'Flush', False, 'Standard', True): ('QO124L125PQG', 'QOC24UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 125, 24, True, 'Surface', False, 'Standard', True): ('QO124L125PQG', 'QOC24US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 125, 30, True, 'Combination', False, 'Standard', True): ('QO124L125PQG', 'QOC30U125C'),
                # Main Lugs, 200A
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 20, True, 'Flush', False, 'Standard', True): ('QO120L200PQ', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 20, True, 'Surface', False, 'Standard', True): ('QO120L200PQ', 'QOC30US'),
                # Main Lugs, 225A
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 42, True, 'Flush', False, 'Standard', True): ('QO142L225PQG', 'QOC42UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 42, True, 'Surface', False, 'Standard', True): ('QO142L225PQG', 'QOC42US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 54, True, 'Flush', False, 'Standard', True): ('QO154L225PQG', 'QOC54UF'),
                # Main Lugs, 200A (additional configurations)
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 30, True, 'Flush', False, 'Standard', True): ('QO130L200PQ', 'QOC30UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 30, True, 'Surface', False, 'Standard', True): ('QO130L200PQ', 'QOC30US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 42, True, 'Flush', False, 'Standard', True): ('QO142L200PQ', 'QOC42UF'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 42, True, 'Surface', False, 'Standard', True): ('QO142L200PQ', 'QOC42US'),
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 54, True, 'Flush', False, 'Standard', True): ('QO154L200PQ', 'QOC54UF'),
                # Included Cover, Main Lug, Copper
                # Main Lugs, 125A
                ('QO', 'NEMA1', '1PHASE', 'L', 125, 12, True, 'Included', False, 'Standard', False): 'QO112L125PGC',
                ('QO', 'NEMA1', '1PHASE', 'L', 125, 20, True, 'Included', False, 'Standard', False): 'QO120L125PGC',
                ('QO', 'NEMA1', '1PHASE', 'L', 125, 24, True, 'Included', False, 'Standard', False): 'QO124L125PGC',
                # Main Lugs, 200A
                ('QO', 'NEMA1', '1PHASE', 'L', 200, 30, True, 'Included', False, 'Standard', False): 'QO130L200PGC',
                # Main Lugs, 225A
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 42, True, 'Included', False, 'Standard', False): 'QO142L225PGC',
                ('QO', 'NEMA1', '1PHASE', 'L', 225, 54, True, 'Included', False, 'Standard', False): 'QO154L225PGC',
                # Main Breaker, 100A
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 12, True, 'Included', False, 'Standard', False): 'QO112M100PC',
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 16, True, 'Included', False, 'Standard', False): 'QO116M100PC',
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 20, True, 'Included', False, 'Standard', False): 'QO120M100PC',
                ('QO', 'NEMA1', '1PHASE', 'M', 100, 24, True, 'Included', False, 'Standard', False): 'QO124M100PC',
                # Main Breaker, 150A
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 30, True, 'Included', False, 'Standard', False): 'QO130M150PC',
                ('QO', 'NEMA1', '1PHASE', 'M', 150, 42, True, 'Included', False, 'Standard', False): 'QO142M150PC',
                # Main Breaker, 200A
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 30, True, 'Included', False, 'Standard', False): 'QO130M200PC',
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 40, True, 'Included', False, 'Standard', False): 'QO140M200PC',
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 42, True, 'Included', False, 'Standard', False): 'QO142M200PC',
                ('QO', 'NEMA1', '1PHASE', 'M', 200, 54, True, 'Included', False, 'Standard', False): 'QO154M200PC',
                # PON, Main Lugs, NEMA3R, Copper
                # 125A Main Lugs, NEMA3R
                ('QO', 'NEMA3R', '1PHASE', 'L', 125, 12, True, 'Included', False, 'Standard', False): 'QO112L125PGRB',
                ('QO', 'NEMA3R', '1PHASE', 'L', 125, 16, True, 'Included', False, 'Standard', False): 'QO116L125PGRB',
                ('QO', 'NEMA3R', '1PHASE', 'L', 125, 24, True, 'Included', False, 'Standard', False): 'QO124L125PGRB',
                # 200A Main Lugs, NEMA3R
                ('QO', 'NEMA3R', '1PHASE', 'L', 200, 12, True, 'Included', False, 'Standard', False): 'QO112L200PGRB',
                ('QO', 'NEMA3R', '1PHASE', 'L', 200, 30, True, 'Included', False, 'Standard', False): 'QO130L200PGRB',
                ('QO', 'NEMA3R', '1PHASE', 'L', 200, 40, True, 'Included', False, 'Standard', False): 'QO140L200PGRB',
                # 225A Main Lugs, NEMA3R
                ('QO', 'NEMA3R', '1PHASE', 'L', 225, 42, True, 'Included', False, 'Standard', False): 'QO142L225PGRB',
                # PON, Main Breaker, NEMA3R, Copper
                # 100A Main Breaker, NEMA3R
                ('QO', 'NEMA3R', '1PHASE', 'M', 100, 12, True, 'Included', False, 'Standard', False): 'QO112M100PRB',
                ('QO', 'NEMA3R', '1PHASE', 'M', 100, 16, True, 'Included', False, 'Standard', False): 'QO116M100PRB',
                ('QO', 'NEMA3R', '1PHASE', 'M', 100, 20, True, 'Included', False, 'Standard', False): 'QO120M100PRB',
                ('QO', 'NEMA3R', '1PHASE', 'M', 100, 24, True, 'Included', False, 'Standard', False): 'QO124M100PRB',
                # 125A Main Breaker, NEMA3R
                ('QO', 'NEMA3R', '1PHASE', 'M', 125, 24, True, 'Included', False, 'Standard', False): 'QO124M125PRB',
                # 150A Main Breaker, NEMA3R
                ('QO', 'NEMA3R', '1PHASE', 'M', 150, 20, True, 'Included', False, 'Standard', False): 'QO120M150PRB',
                ('QO', 'NEMA3R', '1PHASE', 'M', 150, 30, True, 'Included', False, 'Standard', False): 'QO130M150PRB',
                # 200A Main Breaker, NEMA3R
                ('QO', 'NEMA3R', '1PHASE', 'M', 200, 20, True, 'Included', False, 'Standard', False): 'QO120M200PRB',
                ('QO', 'NEMA3R', '1PHASE', 'M', 200, 30, True, 'Included', False, 'Standard', False): 'QO130M200PRB',
                ('QO', 'NEMA3R', '1PHASE', 'M', 200, 40, True, 'Included', False, 'Standard', False): 'QO140M200PRB',
                ('QO', 'NEMA3R', '1PHASE', 'M', 200, 42, True, 'Included', False, 'Standard', False): 'QO142M200PRB',
                # 225A Main Breaker, NEMA3R
                ('QO', 'NEMA3R', '1PHASE', 'M', 225, 42, True, 'Included', False, 'Standard', False): 'QO142M225PRB',
                # 3PHASE, NEMA1, Copper, Main Lugs
                # 125A Main Lugs, 3PHASE, NEMA1
                ('QO', 'NEMA1', '3PHASE', 'L', 125, 12, True, 'Flush', False, 'Standard', False): ('QO312L125G', 'QOC16UF'),
                ('QO', 'NEMA1', '3PHASE', 'L', 125, 12, True, 'Surface', False, 'Standard', False): ('QO312L125G', 'QOC16US'),
                ('QO', 'NEMA1', '3PHASE', 'L', 125, 20, True, 'Flush', False, 'Standard', False): ('QO320L125G', 'QOC24UF'),
                ('QO', 'NEMA1', '3PHASE', 'L', 125, 20, True, 'Surface', False, 'Standard', False): ('QO320L125G', 'QOC24US'),
                ('QO', 'NEMA1', '3PHASE', 'L', 125, 24, True, 'Flush', False, 'Standard', False): ('QO324L125G', 'QOC24UF'),
                ('QO', 'NEMA1', '3PHASE', 'L', 125, 24, True, 'Surface', False, 'Standard', False): ('QO324L125G', 'QOC24US'),
                # 200A Main Lugs, 3PHASE, NEMA1
                ('QO', 'NEMA1', '3PHASE', 'L', 200, 18, True, 'Flush', False, 'Standard', False): ('QO318L200G', 'QOC30UF'),
                ('QO', 'NEMA1', '3PHASE', 'L', 200, 18, True, 'Surface', False, 'Standard', False): ('QO318L200G', 'QOC30US'),
                ('QO', 'NEMA1', '3PHASE', 'L', 200, 30, True, 'Flush', False, 'Standard', False): ('QO330L200G', 'QOC30UF'),
                ('QO', 'NEMA1', '3PHASE', 'L', 200, 30, True, 'Surface', False, 'Standard', False): ('QO330L200G', 'QOC30US'),
                # 225A Main Lugs, 3PHASE, NEMA1
                ('QO', 'NEMA1', '3PHASE', 'L', 225, 42, True, 'Flush', False, 'Standard', False): ('QO342L225G', 'QOC42UF'),
                ('QO', 'NEMA1', '3PHASE', 'L', 225, 42, True, 'Surface', False, 'Standard', False): ('QO342L225G', 'QOC42US'),
                # 3PHASE, NEMA1, Copper, Main Breaker
                # 100A Main Breaker, 3PHASE, NEMA1
                ('QO', 'NEMA1', '3PHASE', 'M', 100, 27, True, 'Flush', False, 'Standard', False): ('QO327M100', 'QOC30UF'),
                ('QO', 'NEMA1', '3PHASE', 'M', 100, 27, True, 'Surface', False, 'Standard', False): ('QO327M100', 'QOC30US'),
                # 125A Main Breaker, 3PHASE, NEMA1
                ('QO', 'NEMA1', '3PHASE', 'M', 125, 30, True, 'Flush', False, 'Standard', False): ('QO330MQ125', 'QOC342MQF'),
                ('QO', 'NEMA1', '3PHASE', 'M', 125, 30, True, 'Surface', False, 'Standard', False): ('QO330MQ125', 'QOC342MQS'),
                # 150A Main Breaker, 3PHASE, NEMA1
                ('QO', 'NEMA1', '3PHASE', 'M', 150, 30, True, 'Flush', False, 'Standard', False): ('QO330MQ150', 'QOC342MQF'),
                ('QO', 'NEMA1', '3PHASE', 'M', 150, 30, True, 'Surface', False, 'Standard', False): ('QO330MQ150', 'QOC342MQS'),
                ('QO', 'NEMA1', '3PHASE', 'M', 150, 42, True, 'Flush', False, 'Standard', False): ('QP342MQ150', 'QOC342MQF'),
                ('QO', 'NEMA1', '3PHASE', 'M', 150, 42, True, 'Surface', False, 'Standard', False): ('QP342MQ150', 'QOC342MQS'),
                # 200A Main Breaker, 3PHASE, NEMA1
                ('QO', 'NEMA1', '3PHASE', 'M', 200, 30, True, 'Flush', False, 'Standard', False): ('QO330MQ200', 'QOC342MQF'),
                ('QO', 'NEMA1', '3PHASE', 'M', 200, 30, True, 'Surface', False, 'Standard', False): ('QO330MQ200', 'QOC342MQS'),
                ('QO', 'NEMA1', '3PHASE', 'M', 200, 42, True, 'Flush', False, 'Standard', False): ('QO342MQ200', 'QOC342MQF'),
                ('QO', 'NEMA1', '3PHASE', 'M', 200, 42, True, 'Surface', False, 'Standard', False): ('QO342MQ200', 'QOC342MQS'),
                # 225A Main Breaker, 3PHASE, NEMA1
                ('QO', 'NEMA1', '3PHASE', 'M', 225, 42, True, 'Flush', False, 'Standard', False): ('QO342MQ225', 'QOC342MQF'),
                ('QO', 'NEMA1', '3PHASE', 'M', 225, 42, True, 'Surface', False, 'Standard', False): ('QO342MQ225', 'QOC342MQS'),
                # 3PHASE, NEMA3R, Copper, Main Lugs and Main Breaker with Included Cover
                # Main Lugs, NEMA3R, Copper, with Included Cover
                ('QO', 'NEMA3R', '3PHASE', 'L', 125, 12, True, 'Included', False, 'Standard', False): 'QO312L125GRB',
                ('QO', 'NEMA3R', '3PHASE', 'L', 125, 20, True, 'Included', False, 'Standard', False): 'QO320L125GRB',
                ('QO', 'NEMA3R', '3PHASE', 'L', 200, 18, True, 'Included', False, 'Standard', False): 'QO318L200GRB',
                ('QO', 'NEMA3R', '3PHASE', 'L', 200, 30, True, 'Included', False, 'Standard', False): 'QO330L200GRB',
                ('QO', 'NEMA3R', '3PHASE', 'L', 225, 42, True, 'Included', False, 'Standard', False): 'QO342L225GRB',
                # Main Breaker, NEMA3R, Copper, with Inluded Cover
                ('QO', 'NEMA3R', '3PHASE', 'M', 100, 27, True, 'Included', False, 'Standard', False): 'QO327M100RB',
                ('QO', 'NEMA3R', '3PHASE', 'M', 125, 30, True, 'Included', False, 'Standard', False): 'QO330MQ125RB',
                ('QO', 'NEMA3R', '3PHASE', 'M', 150, 30, True, 'Included', False, 'Standard', False): 'QO330MQ150RB',
                ('QO', 'NEMA3R', '3PHASE', 'M', 200, 30, True, 'Included', False, 'Standard', False): 'QO330MQ200RB',
                ('QO', 'NEMA3R', '3PHASE', 'M', 200, 42, True, 'Included', False, 'Standard', False): 'QO342MQ200RB',
                ('QO', 'NEMA3R', '3PHASE', 'M', 225, 42, True, 'Included', False, 'Standard', False): 'QO342MQ225RB',
            }

            # Check if the configuration exists in the allowed list for QO
            configKey = (
                loadCenterType, enclosure, phasing, typeOfMain, mainsRating, poleSpaces,
                plugOnNeutral, coverStyle, valuePack, specialApplication, quikGrip
            )
            
            if configKey in allowedConfigurationsQo:
                value = allowedConfigurationsQo[configKey]
                result = {}

                if isinstance(value, tuple):
                    boxAndInteriorPartNumber, coverPartNumber = value
                    result["Box and Interior"] = boxAndInteriorPartNumber
                    result["Cover"] = coverPartNumber
                    if groundBar == True:
                        groundBarKit = self.generateQoGroundBar(boxAndInteriorPartNumber, True)
                        result.update(groundBarKit)
                else:
                    result["Box and Interior"] = value
                    result["Cover"] = "Included"
                    if groundBar == True:
                        groundBarKit = self.generateQoGroundBar(value, True)
                        result.update(groundBarKit)

                return result

            else:
                return {"Error": "Invalid configuration for QO Load Center"}

# End Loadcenters ^

#------------------------------------------------------

# Start Transformers
class transformer():
    def __init__(self):
        # Initialize EasyOCR reader
        self.bool = False
    # Function to generate part number for transformers
    def generateTransformerPartNumber(self, attributes):
        transformerType = attributes.get("transformerType")
        kva = attributes.get("kva")
        primaryVolts = attributes.get("primaryVolts")
        secondaryVolts = attributes.get("secondaryVolts")
        coreMaterial = attributes.get("coreMaterial")
        temperature = attributes.get("temperature")
        weathershield = attributes.get("weathershield")
        mounting = attributes.get("mounting")

        # --- normalize kVA ---
        try:
            if isinstance(kva, str):
                kva = float(kva)
            if isinstance(kva, float) and kva.is_integer():
                kva = int(kva)
        except (TypeError, ValueError):
            return "Invalid kVA value."

        if transformerType != 'WATCHDOG':
            if temperature in (None, ""):
                pass
        else:
            if temperature in (None, ""):
                return "WATCHDOG requires a temperature of 115 or 80."

        # 1) Enforce numeric volts/temperature
        try:
            primaryVolts = int(primaryVolts) if primaryVolts is not None else None
            secondaryVolts = int(secondaryVolts) if secondaryVolts is not None else None
        except (TypeError, ValueError):
            return "Invalid primary/secondary voltage."

        try:
            if temperature is not None and temperature != "":
                temperature = int(temperature)
        except (TypeError, ValueError):
            return "Invalid temperature value."

        # 2) Normalize material/enclosure string
        coreMaterial = (str(coreMaterial).upper() if coreMaterial is not None else None)

        # 3) Normalize mounting for comparison
        mounting = (str(mounting).upper() if mounting else None)

        # Default Lugs Based on kVA
        lugDefaults = {
            15: ("DASKP100", "DASKGS100"),
            30: ("DASKP100", "DASKGS100"),
            45: ("DASKP250", "DASKGS250"),
            75: ("DASKP250", "DASKGS250"),
            112.5: ("DASKP400", "DASKGS400"),
            150: ("DASKP600", "DASKGS600"),
            225: ("DASKP1000", "DASKGS1000"),
            300: ("DASKP1000", "DASKGS1000"),
            500: (None, "DASKGS2000"),  # No primary lugs
            750: (None, None)  # No lugs
        }
        # Weathershield Defaults
        weathershieldDefaults = {
            15: "7400WS17M",
            30: "7400WS18M19M",
            45: "7400WS18M19M",
            75: "7400WS20M",
            112.5: "7400WS21M",
            150: "7400WS22M",
            225: "7400WS25J",
            300: "7400WS25J",
            500: "7400WS30J",
            750: "7400WS31J"
        }

        # Mounting Defaults (wall, ceiling)
        mountingDefaults = {
            15: ("7400WMB17M", "7400CMB17M"),
            30: ("7400WMB18M19M20M", "7400CMB18M19M20M"),
            45: ("7400WMB18M19M20M", "7400CMB18M19M20M"),
            75: ("7400WMB18M19M20M", "7400CMB18M19M20M"),
            112.5: (None, "7400CMB18M19M20M"),  # No wall mount option
            150: (None, "7400CMB18M19M20M"),  # No wall mount option
            225: (None, None),  # No wall/ceiling mount
            300: (None, None),
            500: (None, None),
            750: (None, None)
        }

        # Extract Default Values
        primaryLugs, secondaryLugs = lugDefaults.get(kva, (None, None))
        weathershieldPart = weathershieldDefaults.get(kva) if weathershield else None
        mountingDefaultsEntry = mountingDefaults.get(kva)
        if mountingDefaultsEntry:
            if mounting == "WALL":
                mountingPart = mountingDefaultsEntry[0]
            elif mounting == "CEILING":
                mountingPart = mountingDefaultsEntry[1]
            else:
                mountingPart = None
        else:
            mountingPart = None

        # Transformer configuration rules based on type
        transformerRules = {
            '3PHASESTANDARD': {
                'temperature': 150,
                'prefixRule': lambda kva: 'EXN' if kva <= 150 else 'EX',
                'suffixRules': {
                    (480, 208, 'ALUMINUM'): {
                        15: '3H', 30: '3H', 45: '3H', 75: '3H', 112.5: '3H', 150: '3H', 225: '3H', 300: '3H', 500: '68H', 750: '68H'
                    },
                    (600, 208, 'ALUMINUM'): {
                        15: '65H', 30: '65H', 45: '65H', 75: '65H', 112.5: '65H', 150: '65H', 225: '65H', 300: '65H', 500: '79H', 750: '79H'
                    },
                    (208, 208, 'ALUMINUM'): {
                        15: '3156H', 30: '3156H', 45: '3156H', 75: '3156H', 112.5: '3156H', 150: '3156H', 225: '221H', 300: '221H', 500: '221H'
                    },
                    (240, 208, 'ALUMINUM'): {
                        15: '3156H', 30: '3156H', 45: '3156H', 75: '3156H', 112.5: '3156H', 150: '3156H', 225: '239H', 300: '239H', 500: '239H'
                    },
                    (480, 208, 'COPPER'): {
                        15: '3HCU', 30: '3HCU', 45: '3HCU', 75: '3HCU', 112.5: '3HCU', 150: '3HCU', 225: '3HCU', 300: '3HCU', 500: '68HCU', 750: '68HCU'
                    },
                    (208, 480, 'ALUMINUM'): {
                        15: '3155H', 30: '3155H', 45: '3155H', 75: '3155H', 112.5: '3155H', 150: '3155H', 225: '212H', 300: '212H', 500: '212H'
                    },
                    (408, 480, 'ALUMINUM'): {
                        15: '1814H', 30: '1814H', 45: '1814H', 75: '1814H', 112.5: '1814H', 150: '1814H', 225: '1814H', 300: '1814H', 500: '76H'
                    },
                    (480, 240, 'ALUMINUM'): {
                        15: '6H', 30: '6H', 45: '6H', 75: '6H', 112.5: '6H', 150: '6H'
                    }
                }
            },
            '3PHASE115DEGREE': {
                'temperature': 115,
                'prefixRule': lambda kva: 'EXN' if kva <= 150 else 'EX',
                'suffixRules': {
                    (480, 208, 'ALUMINUM'): {
                        15: '3HF', 30: '3HF', 45: '3HF', 75: '3HF', 112.5: '3HF', 150: '3HF', 225: '3HF', 300: '3HF', 500: '68HF', 750: '68HF'
                    },
                    (480, 208, 'COPPER'): {
                        15: '3HFCU', 30: '3HFCU', 45: '3HFCU', 75: '3HFCU', 112.5: '3HFCU', 150: '3HFCU', 225: '3HFCU', 300: '3HFCU', 500: '68HFCU', 750: '68HFCU'
                    }
                }
            },
            '3PHASE80DEGREE': {
                'temperature': 80,
                'prefixRule': lambda kva: 'EXN' if kva <= 150 else 'EX',
                'suffixRules': {
                    (480, 208, 'ALUMINUM'): {
                        15: '3HB', 30: '3HB', 45: '3HB', 75: '3HB', 112.5: '3HB', 150: '3HB', 225: '3HB', 300: '68HB', 500: '68HB'
                    },
                    (480, 208, 'COPPER'): {
                        15: '3HBCU', 30: '3HBCU', 45: '3HBCU', 75: '3HBCU', 112.5: '3HBCU', 150: '3HBCU', 225: '3HBCU', 300: '68HBCU', 500: '68HBCU'
                    }
                }
            },
            'K13': {
                'temperature': 150,
                'prefixRule': lambda kva: 'EXN' if kva <= 112.5 else 'EX',
                'suffixRules': {
                    (480, 208, 'ALUMINUM'): {
                        15: '3HNLP', 30: '3HNLP', 45: '3HNLP', 75: '3HNLP', 112.5: '3HNLP', 150: '3HNLP', 225: '3HNLP'
                    },
                    (480, 208, 'COPPER'): {
                        15: '3HCUNLP', 30: '3HCUNLP', 45: '3HCUNLP', 75: '3HCUNLP', 112.5: '3HCUNLP', 150: '3HCUNLP', 225: '3HCUNLP'
                    }
                }
            },
            '1PHASESTANDARD': {
                'temperature': 150,
                'prefixRule': lambda kva: 'EE',
                'suffixRules': {
                    (240, 120, 'ALUMINUM'): {
                        15: '3H', 25: '3H', 37.5: '3H', 50: '3H', 75: '3H', 100: '3H', 167: '3H', 250: '3H', 333: '3H'
                    },
                    (600, 120, 'ALUMINUM'): {
                        15: '3534H', 25: '3534H', 37.5: '3534H', 50: '3534H', 75: '3534H', 100: '3534H', 167: '3534H', 250: '3534H', 333: '3534H'
                    },
                    (208, 120, 'ALUMINUM'): {
                        15: '60H', 25: '60H', 37.5: '60H', 50: '60H', 75: '60H', 100: '60H', 167: '60H'
                    },
                    (277, 120, 'ALUMINUM'): {
                        15: '61H', 25: '61H', 37.5: '61H', 50: '61H', 75: '61H', 100: '61H', 167: '61H'
                    }
                }
            },
            'WATCHDOG': {
                'temperatureOptions': [115, 80],
                'prefixRule': lambda kva: 'EE',
                'suffixRules': {
                    (240, 120, 'ALUMINUM', 115): {
                        15: '3HF', 25: '3HF', 37.5: '3HF', 50: '3HF', 75: '3HF', 100: '3HF'
                    },
                    (240, 120, 'ALUMINUM', 80): {
                        15: '3HB', 25: '3HB', 37.5: '3HB', 50: '3HB', 75: '3HB', 100: '3HB'
                    }
                }
            },
            'RESIN': {
                'temperature': 115,
                'prefixRule': lambda kva, enclosure: '4X' if enclosure == '4X' else '',
                'suffixRules': {
                    (480, 208, '3R'): {
                        3: '2F', 6: '2F', 9: '2F', 15: '2F', 30: '2F'
                    },
                    (480, 208, 'SS'): {
                        3: '2SS', 6: '2SS', 9: '2SS', 15: '2SS', 30: '2SS'
                    },
                    (480, 208, '4X'): {
                        3: '2FSS', 6: '2FSS', 9: '2FSS', 15: '2FSS', 30: '2FSS'
                    },
                    (480, 240, '3R'): {
                        3: '5F', 6: '5F', 9: '5F', 15: '75F', 30: '75F'
                    },
                    (480, 240, 'SS'): {
                        3: '5SS', 6: '5SS', 9: '5SS', 15: '75SS', 30: '75SS'
                    },
                    (480, 240, '4X'): {
                        3: '5FSS', 6: '5FSS', 9: '75FSS', 15: '75FSS', 30: '75FSS'
                    },
                    (240, 120, '3R'): {
                        1: '1F', 1.5: '1F', 2: '1F', 3: '1F', 5: '1F', 7.5: '1F', 10: '1F', 15: '1F', 25: '1F'
                    },
                    (240, 120, 'SS'): {
                        1: '1FSS', 1.5: '1FSS', 2: '1FSS', 3: '1FSS', 5: '1FSS', 7.5: '1FSS', 10: '1FSS', 15: '1FSS', 25: '1FSS'
                    },
                    (240, 120, '4X'): {
                        1: '1FSS', 1.5: '1FSS', 2: '1FSS', 3: '1FSS', 5: '1FSS', 7.5: '1FSS', 10: '1FSS', 15: '1FSS', 25: '1FSS'
                    },
                    (480, 120, '3R'): {
                        1: '40F', 1.5: '40F', 2: '40F', 3: '40F', 5: '40F', 7.5: '40F', 10: '40F', 15: '40F', 25: '40F'
                    },
                    (480, 120, 'SS'): {
                        1: '40FSS', 1.5: '40FSS', 2: '40FSS', 3: '40FSS', 5: '40FSS', 7.5: '40FSS', 10: '40FSS', 15: '40FSS', 25: '40FSS'
                    },
                    (480, 120, '4X'): {
                        1: '40FSS', 1.5: '40FSS', 2: '40FSS', 3: '40FSS', 5: '40FSS', 7.5: '40FSS', 10: '40FSS', 15: '40FSS', 25: '40FSS'
                    },
                    (600, 120, '3R'): {
                        1: '51F', 1.5: '51F', 2: '51F', 3: '4F', 5: '4F', 7.5: '4F', 10: '4F', 15: '4F', 25: '4F'
                    },
                    (600, 120, 'SS'): {
                        1: '51FSS', 1.5: '51FSS', 2: '51FSS', 3: '4FSS', 5: '4FSS', 7.5: '4FSS', 10: '4FSS', 15: '4FSS', 25: '4FSS'
                    },
                    (600, 120, '4X'): {
                        1: '51FSS', 1.5: '51FSS', 2: '51FSS', 3: '4FSS', 5: '4FSS', 7.5: '4FSS', 10: '4FSS', 15: '4FSS', 25: '4FSS'
                    },
                    (208, 120, '3R'): {
                        1: '7F', 1.5: '7F', 2: '7F', 3: '60F', 5: '60F', 7.5: '60F', 10: '60F', 15: '60F', 25: '60F'
                    },
                    (208, 120, 'SS'): {
                        1: '7FSS', 1.5: '7FSS', 2: '7FSS', 3: '60FSS', 5: '60FSS', 7.5: '60FSS', 10: '60FSS', 15: '60FSS', 25: '60FSS'
                    },
                    (208, 120, '4X'): {
                        1: '7FSS', 1.5: '7FSS', 2: '7FSS', 3: '60FSS', 5: '60FSS', 7.5: '60FSS', 10: '60FSS', 15: '60FSS', 25: '60FSS'
                    },
                    (277, 120, '3R'): {
                        1: '8F', 1.5: '8F', 2: '8F', 3: '61F', 5: '61F', 7.5: '61F', 10: '61F', 15: '61F', 25: '61F'
                    },
                    (277, 120, 'SS'): {
                        1: '8FSS', 1.5: '8FSS', 2: '8FSS', 3: '61FSS', 5: '61FSS', 7.5: '61FSS', 10: '61FSS', 15: '61FSS', 25: '61FSS'
                    },
                    (277, 120, '4X'): {
                        1: '8FSS', 1.5: '8FSS', 2: '8FSS', 3: '61FSS', 5: '61FSS', 7.5: '61FSS', 10: '61FSS', 15: '61FSS', 25: '61FSS'
                }
            }
        }
    }

        # Extract specific transformer rules based on type
        transformerConfig = transformerRules.get(transformerType)
        if not transformerConfig:
            return "Invalid transformer type specified."

        # If caller omitted temperature and this family has a fixed one, default it
        if transformerType != 'WATCHDOG' and temperature in (None, ""):
            temperature = transformerConfig.get('temperature')

        # Re-validate temperature using ints (you already converted to int above when present)
        if 'temperatureOptions' in transformerConfig:
            if temperature not in transformerConfig['temperatureOptions']:
                return f"Invalid temperature for {transformerType}."
        else:
            if temperature != transformerConfig['temperature']:
                return f"Only {transformerConfig['temperature']}°C is allowed for {transformerType}."

        # Retrieve the suffix rules
        suffixRules = transformerConfig.get('suffixRules')
        if transformerType == 'WATCHDOG':
            suffixKey = (primaryVolts, secondaryVolts, coreMaterial, temperature)
        else:
            suffixKey = (primaryVolts, secondaryVolts, coreMaterial)

        if not suffixRules or suffixKey not in suffixRules or kva not in suffixRules[suffixKey]:
            return f"Invalid configuration for {transformerType} with {kva} kVA."

        # Determine prefix based on the transformer type-specific prefix rule
        if transformerType == 'RESIN':
            enclosure = coreMaterial
            prefix = transformerConfig['prefixRule'](kva, enclosure)
        else:
            prefix = transformerConfig['prefixRule'](kva)
        
        coreSuffix = suffixRules[suffixKey][kva]
        
        # Handle special cases for kVA
        if kva == 112.5:
            kvaStr = '112'
        elif kva == 37.5:
            kvaStr = '37'
        elif kva == 7.5:
            kvaStr = '7'
        else:
            kvaStr = str(kva)
        
        # Build the part number with the correct prefix and suffix for each transformer type
        if transformerType == 'RESIN':
            # Resin encapsulated transformers
            if coreMaterial == '4X':
                partNumber = f"{prefix}{kvaStr}S{coreSuffix}"
            else:
                partNumber = f"{kvaStr}S{coreSuffix}"
        elif transformerType in ['1PHASESTANDARD', 'WATCHDOG']:
            # 1PHASE and Watchdog transformers
            partNumber = f"{prefix}{kvaStr}S{coreSuffix}"
        else:
            # 3PHASE transformers
            partNumber = f"{prefix}{kvaStr}T{coreSuffix}"

            # Construct output JSON
        output = {
            "Part Number": partNumber}
        if primaryLugs:
            output["Primary Lugs"] = primaryLugs
        if secondaryLugs:
            output["Secondary Lugs"] = secondaryLugs
        if weathershieldPart:
            output["Weathershield"] = weathershieldPart
        if mountingPart:
            output["Mounting Bracket"] = mountingPart

        return output

# End Transformers ^

#------------------------------------------------------

# Start NQ Panelboards
class nqPanelboard():
    # List of strictly allowed configurations for main lugs
    allowedConfigurationsLug = {
        # (Amperage, Spaces, Phase, Voltage)
        # 1PHASE Configurations
        (100, 18, '1PHASE', 120): ('NQ18L1', 26),
        (100, 30, '1PHASE', 120): ('NQ30L1', 32),
        (225, 30, '1PHASE', 120): ('NQ30L2', 32),
        (225, 42, '1PHASE', 120): ('NQ42L2', 38),
        (225, 72, '1PHASE', 120): ('NQ72L2', 44),
        (225, 84, '1PHASE', 120): ('NQ84L2', 50),
        (400, 30, '1PHASE', 120): ('NQ30L4', 50),
        (400, 42, '1PHASE', 120): ('NQ42L4', 50),
        (400, 54, '1PHASE', 120): ('NQ54L4', 56),
        (400, 84, '1PHASE', 120): ('NQ84L4C', 68),
        (600, 30, '1PHASE', 120): ('NQ30L6C', 50),
        (600, 42, '1PHASE', 120): ('NQ42L6C', 50),
        (600, 54, '1PHASE', 120): ('NQ54L6C', 56),
        (600, 84, '1PHASE', 120): ('NQ84L6C', 68),
        # 3PHASE Configurations
        (100, 18, '3PHASE', 208): ('NQ418L1', 26),
        (100, 30, '3PHASE', 208): ('NQ430L1', 32),
        (225, 30, '3PHASE', 208): ('NQ430L2', 32),
        (225, 42, '3PHASE', 208): ('NQ442L2', 38),
        (225, 54, '3PHASE', 208): ('NQ454L2', 38),
        (225, 72, '3PHASE', 208): ('NQ472L2', 44),
        (225, 84, '3PHASE', 208): ('NQ484L2', 50),
        (400, 30, '3PHASE', 208): ('NQ430L4', 50),
        (400, 42, '3PHASE', 208): ('NQ442L4', 50),
        (400, 54, '3PHASE', 208): ('NQ454L4', 56),
        (400, 72, '3PHASE', 208): ('NQ472L4', 62),
        (400, 84, '3PHASE', 208): ('NQ484L4C', 68),
        (600, 30, '3PHASE', 208): ('NQ430L6C', 50),
        (600, 42, '3PHASE', 208): ('NQ442L6C', 50),
        (600, 54, '3PHASE', 208): ('NQ454L6C', 56),
        (600, 84, '3PHASE', 208): ('NQ484L6C', 68),
}

    # List of strictly allowed configurations for main breaker panels
    allowedConfigurationsBreaker = {
        # Main breaker configurations with amperage ranges (Min Amperage, Max Amperage, Spaces, Phase, Voltage)
        # 15-100A (Allowed MCCB frames: HD, HG, HJ, HL, HR)
        (15, 100, 18, '1PHASE', 120): ('NQ18L1', 38, ['HD', 'HG', 'HJ', 'HL', 'HR'], 'NQMB2HJ', 'NQHJQLLC'),
        (15, 100, 30, '1PHASE', 120): ('NQ30L1', 44, ['HD', 'HG', 'HJ', 'HL', 'HR'], 'NQMB2HJ', 'NQHJQLLC'),
        (15, 225, 30, '1PHASE', 120): ('NQ30L2', 44, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        (15, 225, 42, '1PHASE', 120): ('NQ42L2', 50, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        (15, 225, 72, '1PHASE', 120): ('NQ72L2', 56, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        (15, 225, 84, '1PHASE', 120): ('NQ84L2', 62, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        # 125-400A (Allowed MCCB frames: LA, LH)
        (125, 400, 30, '1PHASE', 120): ('NQ30L4', 62, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        (125, 400, 42, '1PHASE', 120): ('NQ42L4', 62, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        (125, 400, 54, '1PHASE', 120): ('NQ54L4', 68, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        (125, 400, 84, '1PHASE', 120): ('NQ84L4C', 80, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        # 125-600A (Allowed MCCB frames: LG, LJ, LL)
        (125, 600, 30, '1PHASE', 120): ('NQ30L6C', 62, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 600, 42, '1PHASE', 120): ('NQ42L6C', 68, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 600, 54, '1PHASE', 120): ('NQ54L6C', 74, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 600, 84, '1PHASE', 120): ('NQ84L6C', 86, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        # 15-100A, 3PHASE (Allowed MCCB frames: HD, HG, HJ, HL, HR)
        (15, 100, 18, '3PHASE', 208): ('NQ418L1', 38, ['HD', 'HG', 'HJ', 'HL', 'HR'], 'NQMB2HJ', 'NQHJQLLC'),
        (15, 100, 30, '3PHASE', 208): ('NQ430L1', 44, ['HD', 'HG', 'HJ', 'HL', 'HR'], 'NQMB2HJ', 'NQHJQLLC'),
        # 15-225A, 3PHASE (Allowed MCCB frames: HD, HG, HJ, HL, HR, JD, JG, JJ, JL, JR, QB, QD, QG, QJ)
        (15, 225, 30, '3PHASE', 208): ('NQ430L2', 44, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        (15, 225, 42, '3PHASE', 208): ('NQ442L2', 50, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        (15, 225, 54, '3PHASE', 208): ('NQ454L2', 50, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        (15, 225, 72, '3PHASE', 208): ('NQ472L2', 56, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        (15, 225, 84, '3PHASE', 208): ('NQ484L2', 62, ['HD', 'HG', 'HJ', 'HL', 'HR', 'JD', 'JG', 'JJ', 'JL', 'JR', 'QB', 'QD', 'QG', 'QJ'], 'NQMB2HJ', 'NQMB2Q', 'NQHJQLLC'),
        # 125-400A, 3PHASE (Allowed MCCB frames: LA, LH)
        (125, 400, 30, '3PHASE', 208): ('NQ430L4', 62, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        (125, 400, 42, '3PHASE', 208): ('NQ442L4', 62, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        (125, 400, 54, '3PHASE', 208): ('NQ454L4', 68, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        (125, 400, 72, '3PHASE', 208): ('NQ472L4', 74, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        (125, 400, 84, '3PHASE', 208): ('NQ484L4C', 80, ['LA', 'LH'], 'NQMB4LA', 'NQLALLC'),
        # 125-400A, 3PHASE (Allowed MCCB frames: LG, LJ, LL)
        (125, 400, 30, '3PHASE', 208): ('NQ430L4', 62, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 400, 42, '3PHASE', 208): ('NQ442L4', 68, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 400, 54, '3PHASE', 208): ('NQ454L4', 74, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 400, 72, '3PHASE', 208): ('NQ472L4', 80, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 400, 84, '3PHASE', 208): ('NQ484L4C', 86, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        # 125-600A, 3PHASE (Allowed MCCB frames: LG, LJ, LL)
        (125, 600, 30, '3PHASE', 208): ('NQ430L6C', 62, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 600, 42, '3PHASE', 208): ('NQ442L6C', 68, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 600, 54, '3PHASE', 208): ('NQ454L6C', 74, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 600, 72, '3PHASE', 208): ('NQ472L6C', 80, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
        (125, 600, 84, '3PHASE', 208): ('NQ484L6C', 86, ['LG', 'LJ', 'LL'], 'NQMB6PPL', 'NQPPLLLC'),
}
    
    def __init__(self):
        # Initialize EasyOCR reader
        self.bool = False

    def generateNqPanelboardPartNumber(self, attributes):
        amperage = int(attributes.get("amperage"))
        spaces = int(attributes.get("spaces"))
        typeOfMain = attributes.get("typeOfMain", "MAIN LUG").upper()
        material = attributes.get("material", "ALUMINUM").upper()
        enclosure = attributes.get("enclosure", "NEMA1").upper()
        trimStyle = attributes.get("trimStyle", "").upper()
        voltage = int(str(attributes.get("voltage", "240")).replace("V", ""))

        # infer phase from voltage (120 V → single-phase; everything else → 3-phase)
        phase = "1PHASE" if voltage == 120 else "3PHASE"

        # normalize for lookup: treat 240 V on 3PHASE like 208 V
        lookup_voltage = voltage
        if phase == '3PHASE' and voltage == 240:
            lookup_voltage = 208
        spd_raw = attributes.get("spd", "")
        try:
            spd = int(str(spd_raw).strip().rstrip("kK").rstrip("A")) if str(spd_raw).strip() != "" else None
        except ValueError:
            spd = None
        feedThruLugs = attributes.get("feedThruLugs", False)

        # Mappings for neutral part numbers
        neutralMap = {
            "100%": {100: "NQN1CU", 225: "NQN2CU", 400: "NQN6CU", 600: "NQN6CU"},
            "200%": {100: "NQNL1", 225: "NQNL2", 400: "NQNL4", 600: None}
        }

        mappings = {
            'typeMap': {'MAIN LUG': 'L', 'MAIN BREAKER': 'M'},
            'materialMap': {'ALUMINUM': '', 'COPPER': 'C'},
            'enclosureMap': {'NEMA1': '', 'NEMA3R': 'WP'},
            'trimStyleMap': {'FLUSH': 'F', 'SURFACE': 'S', 'HINGED FLUSH': 'FHR', 'HINGED SURFACE': 'SHR'}
        }

        # SPD mappings
        spdMappings = {
            '1PHASE': {80: 'SSP01SBA08D', 100: 'SSP01SBA10D', 120: 'SSP01SBA12D', 160: 'SSP01SBA16D', 200: 'SSP01SBA20D', 240: 'SSP01SBA24D'},
            '3PHASE': {80: 'SSP02SBA08D', 100: 'SSP02SBA10D', 120: 'SSP02SBA12D', 160: 'SSP02SBA16D', 200: 'SSP02SBA20D', 240: 'SSP02SBA24D'}
        }

        # Always define mccb_frames and hidden‐kits to avoid errors
        mccbFrames              = []
        _hiddenMainBreakerKits  = []
        barrierKit              = None
        serviceEntranceKit      = None

        #   Choose the appropriate allowed configurations based on type_of_main
        configs = (self.allowedConfigurationsLug
                   if typeOfMain == 'MAIN LUG'
                   else self.allowedConfigurationsBreaker)

        if typeOfMain == 'MAIN LUG':
            key = (amperage, spaces, phase, lookup_voltage)
            if key not in configs:
                return f"Invalid NQ LUG configuration: {amperage}A, {spaces} spaces, {phase}, {lookup_voltage}V"
            interiorPartNumber, boxNumber = configs[key]
            mccbFrames = None
            barrierKit = None

        # Else, if Main Breaker was selected, then require a valid main_breaker_type.
        elif typeOfMain == 'MAIN BREAKER':
            match = next(
                k for k in configs
                if k[0] <= amperage <= k[1]
                and k[2] == spaces
                and k[3] == phase
                and k[4] == lookup_voltage
            ) if configs else None

            if not match:
                return f"Invalid NQ BREAKER configuration: {amperage}A, {spaces} spaces, {phase}, {lookup_voltage}V"

            # unpack interior, box, allowed frames, and kits
            parts = configs[match]
            interiorPartNumber, boxNumber, allowedMainBreakers = parts[0], parts[1], parts[2]

            # unpack the two‐step kits list
            _hiddenMainBreakerKits = parts[3:-1]
            barrierKit             = parts[-1]
            # still expose Frames for your rules engine
            mccbFrames = allowedMainBreakers

        else:
            return "Invalid type of main."

        # Add material suffix
        if material == 'COPPER':
            interiorPartNumber += mappings['materialMap'][material]

        # Determine SPD part number if specified
        spdPartNumber = None
        if spd is not None:
            spdPartNumber = spdMappings[phase].get(spd)
            if not spdPartNumber:
                allowed = ", ".join(str(k) for k in sorted(spdMappings[phase].keys()))
                return f"Invalid SPD rating '{spd_raw}' for {phase} NQ Panelboard. Allowed options: {allowed}."

        # Determine trim part number for NEMA1 and NEMA3R
        if enclosure == 'NEMA3R':
            # NEMA3R enclosure doesn't require trim style input; uses WP suffix
            trimPartNumber = f"MH{boxNumber}{mappings['enclosureMap'][enclosure]}"
        else:
            # For NEMA1, determine trim based on trim style and amperage
            if trimStyle == '':
                return "Trim style must be specified for non-NEMA3R enclosures."

            if amperage >= 400:
                trimPartNumber = f"NC{boxNumber}V{mappings['trimStyleMap'][trimStyle]}"
            else:
                trimPartNumber = f"NC{boxNumber}{mappings['trimStyleMap'][trimStyle]}"

        # Return the part numbers for the interior, box, trim, main breaker kit, MCCB frames, and SPD (if applicable)
        output = {
            "Interior": interiorPartNumber,
            "Box": f"MH{boxNumber}BE",
            "Trim": trimPartNumber
        }

        if spdPartNumber:
            output["SPD"] = spdPartNumber

        # --- ALWAYS INCLUDE NEUTRAL AND GROUND BAR ---
        output["Ground Bar Kit"] = "PK27GTACU" if material == "COPPER" else "PK27GTA"
        
        # bump amperage up to the next available neutral size
        def bump_amp(size_map, target):
            sizes = sorted(size_map.keys())
            for s in sizes:
                if s >= target:
                    return size_map[s]
            return "n/a"
        # pick the correct neutral for 100% and 200%
        output["Neutral (100%)"] = bump_amp(neutralMap["100%"], amperage)
        output["Neutral (200%)"] = bump_amp({k: v for k, v in neutralMap["200%"].items() if v}, amperage)

        # --- SERVICE‐ENTRANCE OUTPUT ---
        if serviceEntranceKit:
            output["Service Entrance Note"] = ("For service‐entrance panels: 100% neutral, service entrance kit, and ground bar required")
            output["Service Entrance Kit"] = serviceEntranceKit
        else:
            output["Service Entrance Note"] = "Panel not suitable for service entrance."

        # Feed‐Thru Lugs
        if feedThruLugs:
            if amperage <= 225:
                output["Feed-Thru Lugs"] = "NQFTL2L" if spaces in [30,42] else "NQFTL2H"
            elif amperage == 400:
                output["Feed-Thru Lugs"] = "NQFTL4L" if spaces in [30,42] else "NQFTL4H"

        if mccbFrames:
            output["Allowed MCCB Frames"] = mccbFrames

        # reveal barrier‐kit only if we set one, keep main‐breaker kits hidden for rules‐engine
        if barrierKit:
            output["Service Entrance Barrier Kit"] = barrierKit
        if _hiddenMainBreakerKits:
            output["_hiddenMainBreakerKits"] = _hiddenMainBreakerKits

        return output
    
# End NQ Panelboards ^

# -------------------------------------------------------

# Start NF Panelboards
class nfPanelboard():
    allowedConfigurationsLug = {
        # (Amperage, Spaces, Phase, Voltage)
        (125, 18, '3PHASE', (208, 240, 480)): ('NF418L1', 26),
        (125, 30, '3PHASE', (208, 240, 480)): ('NF430L1', 32),
        (125, 42, '3PHASE', (208, 240, 480)): ('NF442L1C', 38),
        (125, 54, '3PHASE', (208, 240, 480)): ('NF454L1C', 44),
        (250, 30, '3PHASE', (208, 240, 480)): ('NF430L2', 38),
        (250, 42, '3PHASE', (208, 240, 480)): ('NF442L2', 44),
        (250, 54, '3PHASE', (208, 240, 480)): ('NF454L2', 50),
        (250, 66, '3PHASE', (208, 240, 480)): ('NF466L2', 62),
        (400, 30, '3PHASE', (208, 240, 480)): ('NF430L4', 50),
        (400, 42, '3PHASE', (208, 240, 480)): ('NF442L4', 56),
        (400, 54, '3PHASE', (208, 240, 480)): ('NF454L4', 62),
        (400, 66, '3PHASE', (208, 240, 480)): ('NF466L4', 74),
        (400, 84, '3PHASE', (208, 240, 480)): ('NF484L4', 86),
        (600, 30, '3PHASE', (208, 240, 480)): ('NF430L6C', 50),
        (600, 42, '3PHASE', (208, 240, 480)): ('NF442L6C', 56),
        (600, 54, '3PHASE', (208, 240, 480)): ('NF454L6C', 62),
        (600, 66, '3PHASE', (208, 240, 480)): ('NF466L6C', 74),
        (600, 84, '3PHASE', (208, 240, 480)): ('NF484L6C', 86),
    }

    allowedConfigurationsBreaker = {
        # (Min Amps, Max Amps, Spaces, Phase, Voltage)
        # 15-125A (HD, HG, HJ, HL, HR)
        (15, 125, 18, '3PHASE', (208, 240, 480)): ('NF418L1', 38, ['HD', 'HG', 'HJ', 'HL', 'HR'], 'N150MH', 'NFHJLLC'),
        (15, 125, 30, '3PHASE', (208, 240, 480)): ('NF430L1', 44, ['HD', 'HG', 'HJ', 'HL', 'HR'], 'N150MH', 'NFHJLLC'),
        (15, 125, 42, '3PHASE', (208, 240, 480)): ('NF442L1C', 50, ['HD', 'HG', 'HJ', 'HL', 'HR'], 'N150MH', 'NFHJLLC'),
        (15, 125, 54, '3PHASE', (208, 240, 480)): ('NF454L1C', 56, ['HD', 'HG', 'HJ', 'HL', 'HR'], 'N150MH', 'NFHJLLC'),
        # 125-250A (JD, JG, JJ, JL, JR)
        (125, 250, 30, '3PHASE', (208, 240, 480)): ('NF430L2', 50, ['JD', 'JG', 'JJ', 'JL', 'JR'], 'N250MJ', 'NFHJLLC'),
        (125, 250, 42, '3PHASE', (208, 240, 480)): ('NF442L2', 56, ['JD', 'JG', 'JJ', 'JL', 'JR'], 'N250MJ', 'NFHJLLC'),
        (125, 250, 54, '3PHASE', (208, 240, 480)): ('NF454L2', 62, ['JD', 'JG', 'JJ', 'JL', 'JR'], 'N250MJ', 'NFHJLLC'),
        (125, 250, 66, '3PHASE', (208, 240, 480)): ('NF466L2', 74, ['JD', 'JG', 'JJ', 'JL', 'JR'], 'N250MJ', 'NFHJLLC'),
        # 125-400A (LA, LH)
        (125, 400, 30, '3PHASE', (208, 240, 480)): ('NF430L4', 62, ['LA', 'LH'], 'N400M', 'NFLALLC'),
        (125, 400, 42, '3PHASE', (208, 240, 480)): ('NF442L4', 68, ['LA', 'LH'], 'N400M', 'NFLALLC'),
        (125, 400, 54, '3PHASE', (208, 240, 480)): ('NF454L4', 74, ['LA', 'LH'], 'N400M', 'NFLALLC'),
        (125, 400, 66, '3PHASE', (208, 240, 480)): ('NF466L4', 86, ['LA', 'LH'], 'N400M', 'NFLALLC'),
        # 125-600A (LG, LJ, LL, LR)
        (125, 600, 30, '3PHASE', (208, 240, 480)): ('NF430L6C', 68, ['LG', 'LJ', 'LL', 'LR'], 'N600MPPL', 'NFPPLLLC'),
        (125, 600, 42, '3PHASE', (208, 240, 480)): ('NF442L6C', 74, ['LG', 'LJ', 'LL', 'LR'], 'N600MPPL', 'NFPPLLLC'),
        (125, 600, 54, '3PHASE', (208, 240, 480)): ('NF454L6C', 80, ['LG', 'LJ', 'LL', 'LR'], 'N600MPPL', 'NFPPLLLC'),
        (125, 600, 66, '3PHASE', (208, 240, 480)): ('NF466L6C', 86, ['LG', 'LJ', 'LL', 'LR'], 'N600MPPL', 'NFPPLLLC'),
    }

    def __init__(self):
        self.bool = False

    # Helper to bump amperage up to next available neutral
    def _bump_neutral(self, size_map, target):
        sizes = sorted(size_map.keys())
        for s in sizes:
            if s >= target and size_map[s]:
                return size_map[s]
        return "n/a"

    # Helper function to get NF neutral part number
    def getNfNeutralPartNumber(self, amperage, isMainLug, neutral, serviceEntrance, material):
            # Restriction: No 200% neutral allowed for 400A, and no 100% copper neutral for 600A
        if neutral and str(neutral).replace('%', '').strip() == "200" and amperage == 400:
            return "ERROR: 200% neutral is not allowed for 400A NF panelboards."
        if neutral and str(neutral).replace('%', '').strip() == "100" and amperage == 600 and material == "COPPER":
            return "ERROR: 100% copper neutral is not allowed for 600A NF panelboards."
        if serviceEntrance == True:
            return None  # Standard 100% aluminum neutral included
        normalized = str(neutral).replace('%', '').strip()
        if normalized == "100":
            if material == "ALUMINUM":
                return None  # 100% aluminum is standard, no part needed
            if amperage <= 125:
                return "NFN1CU"
            elif amperage <= 250:
                return "NFN2CU"
            elif amperage <= 400:
                return "NFN6CU"
            elif amperage <= 600 and isMainLug:
                return "NFN6CU"
            else:
                return "ERROR: No 100% copper neutral available"
        elif normalized == "200":
            if amperage <= 125:
                return "NFNL1"
            elif amperage <= 250:
                return "NFNL2"
            elif amperage <= 400:
                return "NFNL4"
            else:
                return "ERROR: 200% neutral is not available"
        elif neutral:
            return "ERROR: Invalid neutral input"
        return None

    # Main NF part number logic
    def generateNfPanelboardPartNumber(self, attributes):
        amperage = attributes.get("amperage")
        spaces = attributes.get("spaces")
        typeOfMain = attributes.get("typeOfMain", "MAIN LUG").upper()
        material = attributes.get("material", "ALUMINUM").upper()
        enclosure = attributes.get("enclosure", "NEMA1").upper()
        trimStyle = attributes.get("trimStyle", "")
        trimStyle = trimStyle.upper()
        phase = attributes.get("phase", "3PHASE")
        voltage = int(attributes.get("voltage", 208))
        mainBreakerType = attributes.get("mainBreakerType", "")
        feedThruLugs = attributes.get("feedThruLugs", False)

        breakerKit = None
        serviceEntranceKit = None

        mappings = {
            'typeMap': {'MAIN LUG': 'L', 'MAIN BREAKER': 'M'},
            'materialMap': {'ALUMINUM': '', 'COPPER': 'C'},
            'enclosureMap': {'NEMA1': '', 'NEMA3R': 'WP'},
            'trimStyleMap': {'FLUSH': 'F', 'SURFACE': 'S', 'HINGED FLUSH': 'FHR', 'HINGED SURFACE': 'SHR'},
            'ftlMap': {  
                # Feed-Thru Lug mappings
                125: 'NF125FTL',
                250: 'NF250FTL',
                400: 'NF400FTL',
            }
        }

        # Ground bar kit mappings
        groundBarKits = {
            'ALUMINUM': {
                (125, 18): 'PK12GTA',
                (125, 30): 'PK18GTA',
                (125, 42): 'PK23GTA',
                (125, 54): 'PK23GTA',
                (250, 'any'): 'PK27GTA',
                (400, 'any'): 'PK27GTA',
                (600, 'any'): 'PK27GTA',
            },
            'COPPER': {
                ('any', 'any'): 'PK27GTACU'
            }
        }

        # Validate voltage
        validVoltages = [208, 240, 480]
        if voltage not in validVoltages:
            return f"Invalid voltage '{voltage}'. Allowed options: {', '.join(map(str, validVoltages))}."

        # Determine the appropriate FTL part number if applicable
        ftlPartNumber = None
        if feedThruLugs:
            if amperage <= 125:
                ftlPartNumber = mappings['ftlMap'][125]
            elif 125 < amperage <= 250:
                ftlPartNumber = mappings['ftlMap'][250]
            elif 250 < amperage <= 400:
                ftlPartNumber = mappings['ftlMap'][400]
            else:
                return "Feed-thru lugs are not available for configurations above 400A."
        
        # Extract ground bar kit
        if material == 'COPPER':
            groundBarKit = groundBarKits['COPPER'][('any', 'any')]
        else:
            key = (amperage, spaces) if (amperage, spaces) in groundBarKits['ALUMINUM'] else (amperage, 'any')
            groundBarKit = groundBarKits['ALUMINUM'].get(key, "No ground bar kit available for this configuration.")

        # Extract configuration
        if typeOfMain == 'MAIN LUG':
            try:
                configKey = next(
                    key for key in self.allowedConfigurationsLug
                    if key[0] == amperage
                    and key[1] == spaces
                    and key[2] == phase
                    and voltage in key[3]
                )
            except StopIteration:
                return f"Invalid NF LUG configuration: {amperage}A, {spaces} spaces, {phase}, {voltage}V"
            interiorPartNumber, boxNumber = self.allowedConfigurationsLug[configKey]

        elif typeOfMain == 'MAIN BREAKER':
            try:
                configKey = next(
                    key for key in self.allowedConfigurationsBreaker
                    if key[0] <= amperage <= key[1]
                    and key[2] == spaces
                    and key[3] == phase
                    and voltage in key[4]
                )
            except StopIteration:
                return f"Invalid NF BREAKER configuration: {amperage}A, {spaces} spaces, {phase}, {voltage}V"
            interiorPartNumber, boxNumber, allowedMainBreakers, mainBreakerKit, serviceEntranceKit = \
                self.allowedConfigurationsBreaker[configKey]

            if mainBreakerType not in allowedMainBreakers:
                return f"Invalid main breaker type '{mainBreakerType}'."
            breakerKit = mainBreakerKit

        # Only append a copper “C” if the part number doesn’t already end in C
        if material == 'COPPER' and not interiorPartNumber.endswith('C'):
            interiorPartNumber += mappings['materialMap'][material]

        # Determine trim part number for NEMA1 and NEMA3R
        if enclosure == 'NEMA3R':
            # --- special overrides for 600 A NEMA3R trims ---
            if amperage == 600:
                # map spaces → trim-box number for WP
                _600r = {30: 62, 42: 68, 54: 74, 66: 86}
                if spaces == 84:
                    # no 600A×84 NEMA3R option
                    return f"Invalid NF LUG configuration: {amperage}A, {spaces} spaces, {phase}, {voltage}V"
                trimBox = _600r.get(spaces)
                if not trimBox:
                    # fall back if someone slipped in an unsupported size
                    trimBox = boxNumber
                trimPartNumber = f"MH{trimBox}{mappings['enclosureMap'][enclosure]}"
            else:
                # standard NEMA3R
                trimPartNumber = f"MH{boxNumber}{mappings['enclosureMap'][enclosure]}"
        else:
            # For NEMA1, determine trim based on trim style and amperage
            if trimStyle == '':
                return "Trim style must be specified for non-NEMA3R enclosures."

            if amperage >= 400:
                trimPartNumber = f"NC{boxNumber}V{mappings['trimStyleMap'][trimStyle]}"
            else:
                trimPartNumber = f"NC{boxNumber}{mappings['trimStyleMap'][trimStyle]}"
            if enclosure == 'NEMA1' and amperage == 600:
                trimPartNumber += "3PNF"

        # build base output
        output = {
            "Interior": interiorPartNumber,
            "Box":      f"MH{boxNumber}BE",
            "Trim":     trimPartNumber
        }
        if breakerKit:
            output["Main Breaker Kit"] = breakerKit
        if feedThruLugs:
            output["Feed-Thru Lugs"] = ftlPartNumber

        # --- ALWAYS INCLUDE GROUND BAR KIT ---
        output["Ground Bar Kit"] = groundBarKit

        # --- ALWAYS INCLUDE BOTH NEUTRALS ---
        # your original neutralMap from NQ
        neutralMap = {
            100: "NFN1CU", 250: "NFN2CU", 400: "NFN6CU", 600: "NFN6CU"
        }
        neutralMap200 = {100: "NFNL1", 250: "NFNL2", 400: "NFNL4"}

        output["Neutral (100%)"] = self._bump_neutral(neutralMap, amperage)
        output["Neutral (200%)"] = self._bump_neutral(neutralMap200, amperage)

        # --- SERVICE‐ENTRANCE OUTPUT ---
        if serviceEntranceKit:
            output["Service Entrance Note"] = ("For service‐entrance panels: 100% neutral, service entrance kit, and ground bar required")            
            output["Service Entrance Kit"] = serviceEntranceKit
        else:
            output["Service Entrance Note"] = "Panel not suitable for service entrance."

        return output

# End NF Panelboards ^

#-----------------------------------------------------

# Start I-Line Panelboards
class iLinePanelboard():

    HCJ_configs = {
        ('MAIN LUG', 400, 27, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ14484', 'HCM48TF', 'HC3248DB9'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ14484', 'HCM48TS', 'HC3248DB9'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ14484', 'HCM48TFD', 'HC3248DB9'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ14484', 'HCM48TSD', 'HC3248DB9'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ14486', 'HCM48TF', 'HC3248DB9'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ14486', 'HCM48TS', 'HC3248DB9'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ14486', 'HCM48TFD', 'HC3248DB9'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ14486', 'HCM48TSD', 'HC3248DB9'),
        ('MAIN LUG', 800, 27, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ14488', 'HCM48TF', 'HC3248DB9'),
        ('MAIN LUG', 800, 27, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ14488', 'HCM48TS', 'HC3248DB9'),
        ('MAIN LUG', 800, 27, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ14488', 'HCM48TFD', 'HC3248DB9'),
        ('MAIN LUG', 800, 27, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ14488', 'HCM48TSD', 'HC3248DB9'),
        # NEMA3R - No trim, only box and interior
        ('MAIN LUG', 400, 27, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ14484', None, 'HCJ3248WP'),
        ('MAIN LUG', 600, 27, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ14486', None, 'HCJ3248WP'),
        ('MAIN LUG', 800, 27, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ14488', None, 'HCJ3248WP'),
        # NEMA1 - All trim and box combinations for 45 spaces
        ('MAIN LUG', 400, 45, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ23734', 'HCM73TF', 'HC3273DB9'),
        ('MAIN LUG', 400, 45, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ23734', 'HCM73TS', 'HC3273DB9'),
        ('MAIN LUG', 400, 45, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ23734', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN LUG', 400, 45, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ23734', 'HCM73TSD', 'HC3273DB9'),
        ('MAIN LUG', 600, 45, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ23736', 'HCM73TF', 'HC3273DB9'),
        ('MAIN LUG', 600, 45, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ23736', 'HCM73TS', 'HC3273DB9'),
        ('MAIN LUG', 600, 45, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ23736', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN LUG', 600, 45, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ23736', 'HCM73TSD', 'HC3273DB9'),
        ('MAIN LUG', 800, 45, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ23738', 'HCM73TF', 'HC3273DB9'),
        ('MAIN LUG', 800, 45, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ23738', 'HCM73TS', 'HC3273DB9'),
        ('MAIN LUG', 800, 45, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ23738', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN LUG', 800, 45, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ23738', 'HCM73TSD', 'HC3273DB9'),
        # NEMA3R - No trim, only box and interior
        ('MAIN LUG', 400, 45, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ23734', None, 'HCJ3273WP'),
        ('MAIN LUG', 600, 45, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ23736', None, 'HCJ3273WP'),
        ('MAIN LUG', 800, 45, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ23738', None, 'HCJ3273WP'),
        # NEMA1 - All trim and box combinations for 63 spaces
        ('MAIN LUG', 400, 63, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ32734', 'HCM73TF', 'HC3273DB9'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ32734', 'HCM73TS', 'HC3273DB9'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ32734', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ32734', 'HCM73TSD', 'HC3273DB9'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ32736', 'HCM73TF', 'HC3273DB9'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ32736', 'HCM73TS', 'HC3273DB9'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ32736', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ32736', 'HCM73TSD', 'HC3273DB9'),
        ('MAIN LUG', 800, 63, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ32738', 'HCM73TF', 'HC3273DB9'),
        ('MAIN LUG', 800, 63, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ32738', 'HCM73TS', 'HC3273DB9'),
        ('MAIN LUG', 800, 63, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ32738', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN LUG', 800, 63, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ32738', 'HCM73TSD', 'HC3273DB9'),
        # NEMA3R - No trim, only box and interior
        ('MAIN LUG', 400, 63, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ32734', None, 'HCJ3273WP'),
        ('MAIN LUG', 600, 63, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ32736', None, 'HCJ3273WP'),
        ('MAIN LUG', 800, 63, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ32738', None, 'HCJ3273WP'),
        # Add all 99-space combinations
        ('MAIN LUG', 400, 99, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ50914', 'HCM91TF', 'HC3291DB9'),
        ('MAIN LUG', 400, 99, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ50914', 'HCM91TS', 'HC3291DB9'),
        ('MAIN LUG', 400, 99, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ50914', 'HCM91TFD', 'HC3291DB9'),
        ('MAIN LUG', 400, 99, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ50914', 'HCM91TSD', 'HC3291DB9'),
        ('MAIN LUG', 600, 99, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ50916', 'HCM91TF', 'HC3291DB9'),
        ('MAIN LUG', 600, 99, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ50916', 'HCM91TS', 'HC3291DB9'),
        ('MAIN LUG', 600, 99, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ50916', 'HCM91TFD', 'HC3291DB9'),
        ('MAIN LUG', 600, 99, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ50916', 'HCM91TSD', 'HC3291DB9'),
        ('MAIN LUG', 800, 99, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ50918', 'HCM91TF', 'HC3291DB9'),
        ('MAIN LUG', 800, 99, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ50918', 'HCM91TS', 'HC3291DB9'),
        ('MAIN LUG', 800, 99, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ50918', 'HCM91TFD', 'HC3291DB9'),
        ('MAIN LUG', 800, 99, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ50918', 'HCM91TSD', 'HC3291DB9'),
        # NEMA3R - No trim, only box and interior
        ('MAIN LUG', 400, 99, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ50914', None, 'HCJ3291WP'),
        ('MAIN LUG', 600, 99, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ50916', None, 'HCJ3291WP'),
        ('MAIN LUG', 800, 99, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ50918', None, 'HCJ3291WP'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'COPPER', 'FLUSH'): ('HCJ', 'HCJ14484CU', 'HCM48TF', 'HC3248DB9'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'COPPER', 'SURFACE'): ('HCJ', 'HCJ14484CU', 'HCM48TS', 'HC3248DB9'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ14484CU', 'HCM48TFD', 'HC3248DB9'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ14484CU', 'HCM48TSD', 'HC3248DB9'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'COPPER', 'FLUSH'): ('HCJ', 'HCJ14486CU', 'HCM48TF', 'HC3248DB9'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'COPPER', 'SURFACE'): ('HCJ', 'HCJ14486CU', 'HCM48TS', 'HC3248DB9'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ14486CU', 'HCM48TFD', 'HC3248DB9'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ14486CU', 'HCM48TSD', 'HC3248DB9'),
        # NEMA3R - No trim, only box and interior
        ('MAIN LUG', 400, 27, 'NEMA3R', 'COPPER'): ('HCJ', 'HCJ14484CU', None, 'HCJ3248WP'),
        ('MAIN LUG', 600, 27, 'NEMA3R', 'COPPER'): ('HCJ', 'HCJ14486CU', None, 'HCJ3248WP'),
        # NEMA1 - All trim and box combinations for 63 spaces
        ('MAIN LUG', 400, 63, 'NEMA1', 'COPPER', 'FLUSH'): ('HCJ', 'HCJ32734CU', 'HCM73TF', 'HC3273DB9'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'COPPER', 'SURFACE'): ('HCJ', 'HCJ32734CU', 'HCM73TS', 'HC3273DB9'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ32734CU', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ32734CU', 'HCM73TSD', 'HC3273DB9'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'COPPER', 'FLUSH'): ('HCJ', 'HCJ32736CU', 'HCM73TF', 'HC3273DB9'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'COPPER', 'SURFACE'): ('HCJ', 'HCJ32736CU', 'HCM73TS', 'HC3273DB9'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ32736CU', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ32736CU', 'HCM73TSD', 'HC3273DB9'),
        # NEMA3R - No trim, only box and interior
        ('MAIN LUG', 400, 63, 'NEMA3R', 'COPPER'): ('HCJ', 'HCJ32734CU', None, 'HCJ3273WP'),
        ('MAIN LUG', 600, 63, 'NEMA3R', 'COPPER'): ('HCJ', 'HCJ32736CU', None, 'HCJ3273WP'),
    }
    HCPSU_configs = {
        ('MAIN LUG', 800, 54, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP-SU', 'HCP54868SU', 'HC2686TF4AP', 'HC2686DB'),
        ('MAIN LUG', 800, 54, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP-SU', 'HCP54868SU', 'HC2686TS4AP', 'HC2686DB'),
        ('MAIN LUG', 800, 54, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP-SU', 'HCP54868SU', 'HC2686TFHR', 'HC2686DB'),
        ('MAIN LUG', 800, 54, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP-SU', 'HCP54868SU', 'HC2686TSHR', 'HC2686DB'),
        # NEMA3R - No trim, only box and interior for 54 spaces
        ('MAIN LUG', 800, 54, 'NEMA3R', 'ALUMINUM'): ('HCP-SU', 'HCP54868SU', None, 'HC2686WP'),
    }
    HCP_configs = {
        # 27 Spaces
        ('MAIN LUG', 400, 27, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP14504', 'HCW50TF', 'HC4250DB'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP14504', 'HCW50TS', 'HC4250DB'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP14504', 'HCW50TFD', 'HC4250DB'),
        ('MAIN LUG', 400, 27, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP14504', 'HCW50TSD', 'HC4250DB'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP14506', 'HCW50TF', 'HC4250DB'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP14506', 'HCW50TS', 'HC4250DB'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP14506', 'HCW50TFD', 'HC4250DB'),
        ('MAIN LUG', 600, 27, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP14506', 'HCW50TSD', 'HC4250DB'),
        ('MAIN LUG', 800, 27, 'NEMA1', 'COPPER', 'FLUSH'): ('HCP', 'HCP14508', 'HCW50TF', 'HC4250DB'),
        ('MAIN LUG', 800, 27, 'NEMA1', 'COPPER', 'SURFACE'): ('HCP', 'HCP14508', 'HCW50TS', 'HC4250DB'),
        ('MAIN LUG', 800, 27, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCP', 'HCP14508', 'HCW50TFD', 'HC4250DB'),
        ('MAIN LUG', 800, 27, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCP', 'HCP14508', 'HCW50TSD', 'HC4250DB'),
        ('MAIN LUG', 1200, 27, 'NEMA1', 'COPPER', 'FLUSH'): ('HCP', 'HCP145012N', 'HCW50TF', 'HC4250DB'),
        ('MAIN LUG', 1200, 27, 'NEMA1', 'COPPER', 'SURFACE'): ('HCP', 'HCP145012N', 'HCW50TS', 'HC4250DB'),
        ('MAIN LUG', 1200, 27, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCP', 'HCP145012N', 'HCW50TFD', 'HC4250DB'),
        ('MAIN LUG', 1200, 27, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCP', 'HCP145012N', 'HCW50TSD', 'HC4250DB'),
        # 45 Spaces
        ('MAIN LUG', 400, 45, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP23594', 'HCW59TF', 'HC4259DB'),
        ('MAIN LUG', 400, 45, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP23594', 'HCW59TS', 'HC4259DB'),
        ('MAIN LUG', 400, 45, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP23594', 'HCW59TFD', 'HC4259DB'),
        ('MAIN LUG', 400, 45, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP23594', 'HCW59TSD', 'HC4259DB'),
        ('MAIN LUG', 600, 45, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP23596', 'HCW59TF', 'HC4259DB'),
        ('MAIN LUG', 600, 45, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP23596', 'HCW59TS', 'HC4259DB'),
        ('MAIN LUG', 600, 45, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP23596', 'HCW59TFD', 'HC4259DB'),
        ('MAIN LUG', 600, 45, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP23596', 'HCW59TSD', 'HC4259DB'),
        ('MAIN LUG', 800, 45, 'NEMA1', 'COPPER', 'FLUSH'): ('HCP', 'HCP23598', 'HCW59TF', 'HC4259DB'),
        ('MAIN LUG', 800, 45, 'NEMA1', 'COPPER', 'SURFACE'): ('HCP', 'HCP23598', 'HCW59TS', 'HC4259DB'),
        ('MAIN LUG', 800, 45, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCP', 'HCP23598', 'HCW59TFD', 'HC4259DB'),
        ('MAIN LUG', 800, 45, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCP', 'HCP23598', 'HCW59TSD', 'HC4259DB'),
        ('MAIN LUG', 1200, 45, 'NEMA1', 'COPPER', 'FLUSH'): ('HCP', 'HCP235912N', 'HCW59TF', 'HC4259DB'),
        ('MAIN LUG', 1200, 45, 'NEMA1', 'COPPER', 'SURFACE'): ('HCP', 'HCP235912N', 'HCW59TS', 'HC4259DB'),
        ('MAIN LUG', 1200, 45, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCP', 'HCP235912N', 'HCW59TFD', 'HC4259DB'),
        ('MAIN LUG', 1200, 45, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCP', 'HCP235912N', 'HCW59TSD', 'HC4259DB'),
        # 63 Spaces
        ('MAIN LUG', 400, 63, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP32684', 'HCW68TF', 'HC4268DB'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP32684', 'HCW68TS', 'HC4268DB'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP32684', 'HCW68TFD', 'HC4268DB'),
        ('MAIN LUG', 400, 63, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP32684', 'HCW68TSD', 'HC4268DB'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP32686', 'HCW68TF', 'HC4268DB'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP32686', 'HCW68TS', 'HC4268DB'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP32686', 'HCW68TFD', 'HC4268DB'),
        ('MAIN LUG', 600, 63, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP32686', 'HCW68TSD', 'HC4268DB'),
        ('MAIN LUG', 800, 63, 'NEMA1', 'COPPER', 'FLUSH'): ('HCP', 'HCP32688', 'HCW68TF', 'HC4268DB'),
        ('MAIN LUG', 800, 63, 'NEMA1', 'COPPER', 'SURFACE'): ('HCP', 'HCP32688', 'HCW68TS', 'HC4268DB'),
        ('MAIN LUG', 800, 63, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCP', 'HCP32688', 'HCW68TFD', 'HC4268DB'),
        ('MAIN LUG', 800, 63, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCP', 'HCP32688', 'HCW68TSD', 'HC4268DB'),
        ('MAIN LUG', 1200, 63, 'NEMA1', 'COPPER', 'FLUSH'): ('HCP', 'HCP326812N', 'HCW68TF', 'HC4268DB'),
        ('MAIN LUG', 1200, 63, 'NEMA1', 'COPPER', 'SURFACE'): ('HCP', 'HCP326812N', 'HCW68TS', 'HC4268DB'),
        ('MAIN LUG', 1200, 63, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCP', 'HCP326812N', 'HCW68TFD', 'HC4268DB'),
        ('MAIN LUG', 1200, 63, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCP', 'HCP326812N', 'HCW68TSD', 'HC4268DB'),
    # 99 Spaces
        ('MAIN LUG', 400, 99, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP50864', 'HCW86TF', 'HC4286DB'),
        ('MAIN LUG', 400, 99, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP50864', 'HCW86TS', 'HC4286DB'),
        ('MAIN LUG', 400, 99, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP50864', 'HCW86TFD', 'HC4286DB'),
        ('MAIN LUG', 400, 99, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP50864', 'HCW86TSD', 'HC4286DB'),
        ('MAIN LUG', 600, 99, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP50866', 'HCW86TF', 'HC4286DB'),
        ('MAIN LUG', 600, 99, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP50866', 'HCW86TS', 'HC4286DB'),
        ('MAIN LUG', 600, 99, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP50866', 'HCW86TFD', 'HC4286DB'),
        ('MAIN LUG', 600, 99, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP50866', 'HCW86TSD', 'HC4286DB'),
        ('MAIN LUG', 800, 99, 'NEMA1', 'COPPER', 'FLUSH'): ('HCP', 'HCP50868', 'HCW86TF', 'HC4286DB'),
        ('MAIN LUG', 800, 99, 'NEMA1', 'COPPER', 'SURFACE'): ('HCP', 'HCP50868', 'HCW86TS', 'HC4286DB'),
        ('MAIN LUG', 800, 99, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCP', 'HCP50868', 'HCW86TFD', 'HC4286DB'),
        ('MAIN LUG', 800, 99, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCP', 'HCP50868', 'HCW86TSD', 'HC4286DB'),
        ('MAIN LUG', 1200, 99, 'NEMA1', 'COPPER', 'FLUSH'): ('HCP', 'HCP508612N', 'HCW86TF', 'HC4286DB'),
        ('MAIN LUG', 1200, 99, 'NEMA1', 'COPPER', 'SURFACE'): ('HCP', 'HCP508612N', 'HCW86TS', 'HC4286DB'),
        ('MAIN LUG', 1200, 99, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCP', 'HCP508612N', 'HCW86TFD', 'HC4286DB'),
        ('MAIN LUG', 1200, 99, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCP', 'HCP508612N', 'HCW86TSD', 'HC4286DB'),
    }
    HCRU_configs = {
        # HCR-U Main Lug
        ('MAIN LUG', 1200, 108, 'NEMA1', 'COPPER', 'FLUSH'): ('HCR-U', 'HCR548612U', 'HCR86TF', 'HC4486DB'),
        ('MAIN LUG', 1200, 108, 'NEMA1', 'COPPER', 'SURFACE'): ('HCR-U', 'HCR548612U', 'HCR86TS', 'HC4486DB'),
        ('MAIN LUG', 1200, 108, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCR-U', 'HCR548612U', 'HCR86TFD', 'HC4486DB'),
        ('MAIN LUG', 1200, 108, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCR-U', 'HCR548612U', 'HCR86TSD', 'HC4486DB'),
    }

    allowedConfigurationsBreaker = {
        # Service Entrance Rated (if applicable, add Main Breaker and Barrier Kit if needed)
        ('MAIN BREAKER', 800, 54, 'NEMA1', 'ALUMINUM', 'FLUSH', True): ('HCP-SU', 'HCP54868SU', 'HC2686TF4AP', 'HC2686DB'),
        ('MAIN BREAKER', 800, 54, 'NEMA1', 'ALUMINUM', 'SURFACE', True): ('HCP-SU', 'HCP54868SU', 'HC2686TS4AP', 'HC2686DB'),
        ('MAIN BREAKER', 800, 54, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR', True): ('HCP-SU', 'HCP54868SU', 'HC2686TFHR', 'HC2686DB'),
        ('MAIN BREAKER', 800, 54, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR', True): ('HCP-SU', 'HCP54868SU', 'HC2686TSHR', 'HC2686DB'),
        ('MAIN BREAKER', 800, 54, 'NEMA3R', 'ALUMINUM', None, True): ('HCP-SU', 'HCP54868SU', None, 'HC2686WP'),
        # 27 SPACES X 400A X HCJ14734M x HCM73T() X HC3273DB9(NEMA1)
        ('MAIN BREAKER', 400, 27, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ14734M', 'HCM73TF', 'HC3273DB9'),
        ('MAIN BREAKER', 400, 27, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ14734M', 'HCM73TS', 'HC3273DB9'),
        ('MAIN BREAKER', 400, 27, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ14734M', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN BREAKER', 400, 27, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ14734M', 'HCM73TSD', 'HC3273DB9'),
        # NEMA3R
        ('MAIN BREAKER', 400, 27, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ14734M', None, 'HCJ3273WP'),
        # 36 SPACES X 600A X HCJ18736MP x HCM73T() X HC3273DB9
        ('MAIN BREAKER', 600, 36, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ18736MP', 'HCM73TF', 'HC3273DB9'),
        ('MAIN BREAKER', 600, 36, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ18736MP', 'HCM73TS', 'HC3273DB9'),
        ('MAIN BREAKER', 600, 36, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ18736MP', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN BREAKER', 600, 36, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ18736MP', 'HCM73TSD', 'HC3273DB9'),
        ('MAIN BREAKER', 600, 36, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ18736MP', None, 'HCJ3273WP'),
        # 36 SPACES X 800A X HCJ18738MP x HCM73T() X HC3273DB9
        ('MAIN BREAKER', 800, 36, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ18738MP', 'HCM73TF', 'HC3273DB9'),
        ('MAIN BREAKER', 800, 36, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ18738MP', 'HCM73TS', 'HC3273DB9'),
        ('MAIN BREAKER', 800, 36, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ18738MP', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN BREAKER', 800, 36, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ18738MP', 'HCM73TSD', 'HC3273DB9'),
        ('MAIN BREAKER', 800, 36, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ18738MP', None, 'HCJ3273WP'),
        # 45 SPACES X 400A X HCJ23734M x HCM73T() X HC3273DB9
        ('MAIN BREAKER', 400, 45, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ23734M', 'HCM73TF', 'HC3273DB9'),
        ('MAIN BREAKER', 400, 45, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ23734M', 'HCM73TS', 'HC3273DB9'),
        ('MAIN BREAKER', 400, 45, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ23734M', 'HCM73TFD', 'HC3273DB9'),
        ('MAIN BREAKER', 400, 45, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ23734M', 'HCM73TSD', 'HC3273DB9'),
        ('MAIN BREAKER', 400, 45, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ23734M', None, 'HCJ3273WP'),
        # 72 SPACES X 600A X HCJ36916MP x HCM91T() X HC3291DB9
        ('MAIN BREAKER', 600, 72, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ36916MP', 'HCM91TF', 'HC3291DB9'),
        ('MAIN BREAKER', 600, 72, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ36916MP', 'HCM91TS', 'HC3291DB9'),
        ('MAIN BREAKER', 600, 72, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ36916MP', 'HCM91TFD', 'HC3291DB9'),
        ('MAIN BREAKER', 600, 72, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ36916MP', 'HCM91TSD', 'HC3291DB9'),
        ('MAIN BREAKER', 600, 72, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ36916MP', None, 'HCJ3291WP'),
        # 72 SPACES X 800A X HCJ36918MP x HCM91T() X HC3291DB9
        ('MAIN BREAKER', 800, 72, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ36918MP', 'HCM91TF', 'HC3291DB9'),
        ('MAIN BREAKER', 800, 72, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ36918MP', 'HCM91TS', 'HC3291DB9'),
        ('MAIN BREAKER', 800, 72, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ36918MP', 'HCM91TFD', 'HC3291DB9'),
        ('MAIN BREAKER', 800, 72, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ36918MP', 'HCM91TSD', 'HC3291DB9'),
        ('MAIN BREAKER', 800, 72, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ36918MP', None, 'HCJ3291WP'),
        # 82 SPACES X 400A X HCJ41914M (aluminum)
        ('MAIN BREAKER', 400, 82, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCJ', 'HCJ41914M', 'HCM91TF', 'HC3291DB9'),
        ('MAIN BREAKER', 400, 82, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCJ', 'HCJ41914M', 'HCM91TS', 'HC3291DB9'),
        ('MAIN BREAKER', 400, 82, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ41914M', 'HCM91TFD', 'HC3291DB9'),
        ('MAIN BREAKER', 400, 82, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ41914M', 'HCM91TSD', 'HC3291DB9'),
        ('MAIN BREAKER', 400, 82, 'NEMA3R', 'ALUMINUM'): ('HCJ', 'HCJ41914M', None, 'HCJ3291WP'),
        # 82 SPACES X 400A X HCJ41914MCU (copper)
        ('MAIN BREAKER', 400, 82, 'NEMA1', 'COPPER', 'FLUSH'): ('HCJ', 'HCJ41914MCU', 'HCM91TF', 'HC3291DB9'),
        ('MAIN BREAKER', 400, 82, 'NEMA1', 'COPPER', 'SURFACE'): ('HCJ', 'HCJ41914MCU', 'HCM91TS', 'HC3291DB9'),
        ('MAIN BREAKER', 400, 82, 'NEMA1', 'COPPER', 'FLUSH WITH DOOR'): ('HCJ', 'HCJ41914MCU', 'HCM91TFD', 'HC3291DB9'),
        ('MAIN BREAKER', 400, 82, 'NEMA1', 'COPPER', 'SURFACE WITH DOOR'): ('HCJ', 'HCJ41914MCU', 'HCM91TSD', 'HC3291DB9'),
        ('MAIN BREAKER', 400, 82, 'NEMA3R', 'COPPER'): ('HCJ', 'HCJ41914MCU', None, 'HCJ3291WP'),
        # 36 Spaces
        ('MAIN BREAKER', 600, 36, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP18686M', 'HCW68TF', 'HC4268DB'),
        ('MAIN BREAKER', 600, 36, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP18686M', 'HCW68TS', 'HC4268DB'),
        ('MAIN BREAKER', 600, 36, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP18686M', 'HCW68TFD', 'HC4268DB'),
        ('MAIN BREAKER', 600, 36, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP18686M', 'HCW68TSD', 'HC4268DB'),
        ('MAIN BREAKER', 800, 36, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP18688M', 'HCW68TF', 'HC4268DB'),
        ('MAIN BREAKER', 800, 36, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP18688M', 'HCW68TS', 'HC4268DB'),
        ('MAIN BREAKER', 800, 36, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP18688M', 'HCW68TFD', 'HC4268DB'),
        ('MAIN BREAKER', 800, 36, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP18688M', 'HCW68TSD', 'HC4268DB'),
        # 72 Spaces
        ('MAIN BREAKER', 600, 72, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP36866M', 'HCW86TF', 'HC4268DB'),
        ('MAIN BREAKER', 600, 72, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP36866M', 'HCW86TS', 'HC4268DB'),
        ('MAIN BREAKER', 600, 72, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP36866M', 'HCW86TFD', 'HC4268DB'),
        ('MAIN BREAKER', 600, 72, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP36866M', 'HCW86TSD', 'HC4268DB'),
        ('MAIN BREAKER', 800, 72, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCP', 'HCP36868M', 'HCW86TF', 'HC4286DB'),
        ('MAIN BREAKER', 800, 72, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCP', 'HCP36868M', 'HCW86TS', 'HC4286DB'),
        ('MAIN BREAKER', 800, 72, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCP', 'HCP36868M', 'HCW86TFD', 'HC4286DB'),
        ('MAIN BREAKER', 800, 72, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCP', 'HCP36868M', 'HCW86TSD', 'HC4286DB'),
        # HCR-U Main Breaker (Requires selection of MCCB, e.g., PG, PJ, PL, RGC, RJC, RLC)
        ('MAIN BREAKER', 1200, 108, 'NEMA1', 'ALUMINUM', 'FLUSH'): ('HCR-U', 'HCR548612U', 'HCR86TF', 'HC4486DB'),
        ('MAIN BREAKER', 1200, 108, 'NEMA1', 'ALUMINUM', 'SURFACE'): ('HCR-U', 'HCR548612U', 'HCR86TS', 'HC4486DB'),
        ('MAIN BREAKER', 1200, 108, 'NEMA1', 'ALUMINUM', 'FLUSH WITH DOOR'): ('HCR-U', 'HCR548612U', 'HCR86TFD', 'HC4486DB'),
        ('MAIN BREAKER', 1200, 108, 'NEMA1', 'ALUMINUM', 'SURFACE WITH DOOR'): ('HCR-U', 'HCR548612U', 'HCR86TSD', 'HC4486DB'),
    }
        
    def __init__(self):
            # Initialize EasyOCR reader
            self.bool = False

    def generateILinePanelboardPartNumber(self, attributes):
        panelType = attributes.get("panelType")
        amperage = attributes.get("amperage")
        spaces = attributes.get("spaces")
        enclosure = attributes.get("enclosure", "NEMA1")
        material = attributes.get("material", "ALUMINUM")
        trimStyle = attributes.get("trimStyle", "")  # F or S (Flush or Surface)
        typeOfMain = attributes.get("typeOfMain", "MAIN LUG").upper()  # 'MAIN LUG' or 'MAIN BREAKER'
        mainBreakerType = attributes.get("mainBreakerType")
        voltage = int(str(attributes.get("voltage", "240")).replace("V", ""))
        spd_raw = attributes.get("spd", "")
        try:
            spd = int(spd_raw)
        except (TypeError, ValueError):
            spd = None
        
        # SPD Lookup Table
        spdLookup = {
            208: {
                120: 'HL2IMA12C',
                160: 'HL2IMA16C',
                240: 'HL2IMA24C',
            },
            240: {
                240: 'HL6IMA24C',
            },
            480: {
                120: 'HL4IMA12C',
                160: 'HL4IMA16C',
                240: 'HL4IMA24C',
            },
            600: {
                160: 'HR8IMA16C',
                240: 'HR8IMA24C',
            },
        }

        # Barrier kit mappings by panel and breaker type
        barrierKitMap = {
            "HCJ": {
                "H": "ILBFMHCJHULC",
                "J": "ILBFMHCJHULC",
            },
            "HCP": {
                "H": "ILBFMHCPHJULC",
                "J": "ILBFMHCPHJULC",
                "L": "ILBFMHCPLULC",
                "M": "ILBFMHCPMPULC",
                "P": "ILBFMHCPMPULC",
            },
            "HCP-SU": {
                "H": "ILBFMHCPHJULC",
                "J": "ILBFMHCPHJULC",
                "L": "ILBFMHCPLULC",
                "M": "ILBFMHCPMPULC",
                "P": "ILBFMHCPMPULC",
            },
            "HCR-U": {
                "L": "ILBFMHCRLULC",
                "M": "ILBFMHCRMULC",
                "P": "ILBFMHCRPULC",
                "R": "ILBFMHCRRULC",
            },
        }

        # build list of dicts to search, in series priority order
        if typeOfMain == 'MAIN LUG':
            config_sources = [
                self.HCJ_configs,
                self.HCPSU_configs,
                self.HCP_configs,
                self.HCRU_configs,
            ]
        else:
            config_sources = [self.allowedConfigurationsBreaker]

        # --- Build lookup key exactly as in your flat tables ---
        if enclosure == 'NEMA3R':
            # all NEMA3R keys omit the trimStyle element
            key = (typeOfMain, amperage, spaces, enclosure, material)
        elif typeOfMain == 'MAIN BREAKER' and panelType == 'HCP-SU':
            # HCP-SU service-entrance breaker keys include True at the end
            key = (typeOfMain, amperage, spaces, enclosure, material, trimStyle, True)
        else:
            key = (typeOfMain, amperage, spaces, enclosure, material, trimStyle)

        # walk each series-dict until we find a match
        for cfg in config_sources:
            if key in cfg:
                panelTypeLookup, interior, box, trimCode = cfg[key]
                break
        else:
            return f"Invalid I-Line configuration: {key}"

        attributes["panelType"] = panelTypeLookup

        if panelType == "HCJ" and material == 'COPPER' and interior[-1].isdigit():
            interior += 'CU'

        result = {
            "Interior": interior,
            "Box":      (trimCode if enclosure == 'NEMA3R' else box),
            "Trim":     ("Included with NEMA3R" if enclosure=='NEMA3R' else trimCode)
        }

        # --- SPD (unchanged) ---
        if spd is not None:
            spd_map = spdLookup.get(voltage, {})
            result["SPD"] = spd_map.get(spd,
                f"Error: no SPD for {voltage}V/{spd}kA"
            )

        # --- Neutral (always) ---
        def _bump(map_, A):
            for s in sorted(map_):
                if s >= A:
                    return map_[s]
            return "Not available"
        neutralMap = {400:'HCW4SN', 600:'HCW6SN', 800:'HCW8SN', 1200:'HCW12SN'}
        result["Neutral"] = _bump(neutralMap, amperage)

        # --- Determine barrier kit for SE-rated breakers ---
        barrierKit = None
        if typeOfMain == 'MAIN BREAKER' and panelType in barrierKitMap:
            # if we know a breaker‐type letter, use it…
            if mainBreakerType:
                letter = mainBreakerType[:1].upper()
                barrierKit = barrierKitMap[panelType].get(letter)
            # otherwise fall back to the first one in the map
            if not barrierKit:
                barrierKit = next(iter(barrierKitMap[panelType].values()), None)

        # --- Ground bar (always) ---
        result["Ground Bar Kit"] = (
            "PK32DGTA" if material=="ALUMINUM" else "PK32GTACU"
        )

        # --- Service‐Entrance Kit & Note ---
        if barrierKit:
            result["Note"] = ("For service‐entrance I-Line panels: neutral + ground bar + barrier kit required")
            result["Service Entrance Kit"] = barrierKit
        else:
            result["Note"] = "Standard non-SE configuration"
            result["Service Entrance Kit"] = "Panel not suitable for service entrance"

        return result

# End I-Line Panelboards ^

#-----------------------------------------------------

# Start SPD
class externalSpd():
    def __init__(self):
        # Initialize EasyOCR reader
        self.bool = False

    # Function to generate external SPD part numbers
    def generateExternalSpdPartNumber(self, attributes):
        serviceVoltage = attributes.get("serviceVoltage")
        interruptingRating = attributes.get("interruptingRating")

        spdMapping = {
            # 120/240V, 1PH
            (120, 100): "HWB11",
            (120, 200): "EMB12",
            (120, 300): "HWC13",
            (120, 500): "EMD13",
            # 208V, 3PH
            (208, 100): "HWB21",
            (208, 200): "EMB22",
            (208, 300): "HWC23",
            (208, 500): "EMD25",
            (208, 600): "EMD26",
            # 240V, 3PH
            (240, 100): "HWB61",
            (240, 200): "EMB62",
            (240, 300): "HWC63",
            # 480V, 3PH
            (480, 100): "HWB41",
            (480, 200): "EMB42",
            (480, 300): "HWC43",
            (480, 500): "EMD45",
            (480, 600): "EMD46",
            # 480V Delta, 3PH
            ("480DELTA", 100): "HWB51",
            ("480DELTA", 200): "EMB52",
            ("480DELTA", 300): "HWC53",
            ("480DELTA", 500): "EMD55",
            # 600V, 3PH
            (600, 100): "HWB91",
            (600, 200): "HWC93",
            (600, 300): "HWC93",
        }
        key = (serviceVoltage, interruptingRating)
        partNumber = spdMapping.get(key)

        if partNumber:
            return {"Part Number": partNumber}
        else:
            return {"Error": f"Invalid SPD configuration for {serviceVoltage} at {interruptingRating}"}

# End SPD ^

#-----------------------------------------------------

# Start Blanks
import math
class blanks():
    def __init__(self):
        # Initialize EasyOCR reader
        self.bool = False

    # Function to generate Blanks/Filler plates part numbers
    def generateFillerPlatePartNumber(self, attributes):
        panelType = attributes.get("panelType").upper().replace("-", "")
        totalBlankSpaces = attributes.get("totalBlankSpaces")
        
        mapping = {
            'QO':   {'part': 'QOFP',    'packSize': 1},
            'HOM':  {'part': 'HOMFP',   'packSize': 5},
            'NQ':   {'part': 'NQFP15',  'packSize': 15},
            'NF':   {'part': 'NFFP15',  'packSize': 15},
        }
        
        # Validate input
        if panelType not in mapping:
            return {"Error": f"Invalid panel type '{panelType}' for filler plates."}
        
        partInfo = mapping[panelType]
        packSize = partInfo['packSize']
        partNumber = partInfo['part']

        # Calculate how many packs are needed
        packsNeeded = math.ceil(totalBlankSpaces / packSize)

        return {
            "Part Number": partNumber,
            "Quantity": packsNeeded,
            "Covers": f"{packsNeeded * packSize} spaces"
        }

    def generateILineBlanks(self, *, widePieces: int = 0, narrowPieces: int = 0, extensionPieces: int = 0) -> dict:
        """
          - HLW1BL (wide-side blank):    1 piece = 1.5" (1 space). Packs of 3.
          - HLN1BL (narrow-side blank):  1 piece = 1.5" (1 space). Packs of 3.
          - HLW4EBL (wide extension):    1 piece = 4.5" (3 spaces). Packs of 5.
        """
        import math

        items = []

        if widePieces > 0:
            pack_size = 3
            packs = math.ceil(widePieces / pack_size)
            items.append({
                "Part Number": "HLW1BL",
                "Quantity (packs)": packs,
                "Pieces requested": widePieces,
                "Covers (spaces)": packs * pack_size,          # 1 piece = 1 space
                "Note": "Wide-side 1.5\" blanks"
            })

        if narrowPieces > 0:
            pack_size = 3
            packs = math.ceil(narrowPieces / pack_size)
            items.append({
                "Part Number": "HLN1BL",
                "Quantity (packs)": packs,
                "Pieces requested": narrowPieces,
                "Covers (spaces)": packs * pack_size,          # 1 piece = 1 space
                "Note": "Narrow-side 1.5\" blanks"
            })

        if extensionPieces > 0:
            pack_size = 5
            packs = math.ceil(extensionPieces / pack_size)
            # each extension is 4.5" = 3 spaces of coverage
            items.append({
                "Part Number": "HLW4EBL",
                "Quantity (packs)": packs,
                "Pieces requested": extensionPieces,
                "Covers (spaces)": packs * pack_size * 3,      # 1 piece = 3 spaces
                "Note": "Wide-side 4.5\" extensions for H/J on wide side"
            })

        return {"Items": items} if items else {}


# End Blanks ^

#-----------------------------------------------------

# Start MCCB
class mccb():
    def __init__(self):
        # Initialize EasyOCR reader
        self.bool = False

    # Function to generate MCCB part numbers
    def generateMccbPartNumber(self, attributes):
        frame = attributes.get("frame")
        termination = attributes.get("termination")
        poles = attributes.get("poles")
        amperage = attributes.get("amperage")
        voltage = attributes.get("voltage")
        intRating = attributes.get("intRating")
        tripFunction = attributes.get("tripFunction")
        grounding = attributes.get("grounding", False)
        enclosureType = attributes.get("enclosureType", "NONE")
        phasing = attributes.get("phasing", "").upper()
        frame = frame.upper()
        termUpper = termination.upper()

        # I-Line suffix rules (termination must be "ILINE" and tripFunction must be "MAGNETIC")
        suffix = ""
        if termUpper == "ILINE" and tripFunction and tripFunction.upper() == "MAGNETIC":
            if poles == 1:
                suffix_map_1p = {"A": "1", "B": "3", "C": "5"}
                suffix = suffix_map_1p.get(phasing, "")
            elif poles == 2:
                suffix_map_2p = {"AB": "1", "AC": "2", "BA": "3", "BC": "4", "CA": "5", "CB": "6"}
                suffix = suffix_map_2p.get(phasing, "")
            # 3P magnetic gets no suffix change
            # Electronic breakers ignore suffix override

        if frame == "B":
            result = self.generateMccbPartNumberB(termination, poles, amperage, voltage, intRating, suffix)
        elif frame in ["H", "J"]:
            if tripFunction is None:
                return "Trip function must be provided for H and J frames (e.g., 'magnetic' or 'electronic')."
            result = self.generateMccbPartNumberHJ(frame, termination, poles, amperage, voltage, intRating, tripFunction, grounding, suffix)
        elif frame == "Q":
            result = self.generateMccbPartNumberQ(termination, poles, amperage, voltage, intRating, suffix)
        elif frame == "L":
            if tripFunction is None:
                return "Trip function must be provided for L frame breakers (e.g., 'magnetic' or 'electronic')."
            if tripFunction.upper() == "MAGNETIC":
                return self.generateMccbPartNumberLL(termination, poles, amperage, voltage, intRating, suffix)
            elif tripFunction.upper() == "ELECTRONIC":
                result = self.generateMccbPartNumberLElectronic(termination, poles, amperage, voltage, intRating, grounding, suffix)
            else:
                return f"Invalid trip function '{tripFunction}' for L frame breakers. Allowed: magnetic, electronic."
        elif frame == "M":
            if tripFunction is None:
                return "Trip function must be provided for M frame breakers; only 'electronic' is allowed."
            if tripFunction.upper() != "ELECTRONIC":
                return "M frame breakers only allow electronic trip."
            if grounding:
                return "M frame breakers must have grounding set to False."
            result = self.generateMccbPartNumberM(termination, poles, amperage, voltage, intRating, grounding, suffix)
        elif frame == "P":
            if tripFunction is None or tripFunction.upper() != "ELECTRONIC":
                return "P frame breakers only allow electronic trip."
            result = self.generateMccbPartNumberP(termination, poles, amperage, voltage, intRating, grounding, suffix)
        elif frame == "R":
            if tripFunction is None or tripFunction.upper() != "ELECTRONIC":
                return "R frame breakers only allow electronic trip."
            result = self.generateMccbPartNumberR(termination, poles, amperage, voltage, intRating, grounding, suffix=suffix, tripFunction=tripFunction)
        else:
            return f"ERROR: Only Frame type B, H, J, Q, L, M, P, or R allowed"

        if isinstance(result, str) and result.startswith("Invalid") or "must be" in result or "only allow" in result:
            return result

        DENYLIST = {
            "HDA261256",  # remove
            "HLA261001",  # remove
            "HLA261005",  # remove
        }
        candidate = result if isinstance(result, str) else (result.get("Part Number") if isinstance(result, dict) else None)
        if candidate in DENYLIST:
            return f"Invalid part number for current configuration (internal denylist): {candidate}"

        enclosure = self.mccbEnclosure(result, poles, amperage, enclosureType)
        if enclosure:
            return {"Part Number": result, "Enclosure": enclosure}

        return result

    # B Frame Specific Functions
    def getInterruptingLetterB(self, voltage, intRating, poles):
        mapping = {
            240: {25: "D", 65: "G", 100: ["J", "K"]},
            480: {18: "D", 35: "G", 65: ["J", "K"]},
            600: {14: "D", 18: "G", 25: "J", 65: "K"}
        }
        if voltage not in mapping:
            return None
        if intRating not in mapping[voltage]:
            return None
        letter = mapping[voltage][intRating]
        if isinstance(letter, list):
            # For 240V and 480V, always choose "J"
            if voltage in [240, 480]:
                return "J"
            else:
                if poles == 3:
                    return "J"
                elif poles in [1, 2]:
                    return "K"
                else:
                    return None
        else:
            # Additional check: for 600V, if letter is "K", ensure poles are only 1 or 2.
            if voltage == 600 and letter == "K" and poles not in [1, 2]:
                return None
            return letter

    def generateMccbPartNumberB(self, termination, poles, amperage, voltage, intRating, suffix=""):
        # Validate termination input.
        termUpper = termination.upper()
        if termUpper == "LUG":
            termCode = "L"
        elif termUpper == "ILINE":
            termCode = "A"
        else:
            return f"Invalid termination: {termination}. Use 'LUG' or 'ILINE'."

        # Validate poles.
        if poles not in [1, 2, 3]:
            return f"Invalid pole count {poles} for B frame breakers."

        # Determine the interrupting letter based on voltage, int_rating, and poles.
        intrLetter = self.getInterruptingLetterB(voltage, intRating, poles)
        if intrLetter is None:
            return f"Invalid interrupting rating {intRating} kA for voltage {voltage}V and {poles} pole(s)."

        # Set allowed amperages based on the resulting interrupting letter.
        if intrLetter == "K":
            allowedAmperages = [15, 20, 25, 30]
        else:
            allowedAmperages = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 125]
        if amperage not in allowedAmperages:
            return f"Invalid amperage {amperage}A for B frame with {intrLetter} interrupting rating. Allowed: {allowedAmperages}"

        # Validate voltage.
        if voltage < 240:
            return "Invalid voltage for B frame breakers; must be at least 240V."
        voltageCode = "6"

        # Format amperage (if under 100, pad to three digits).
        amperageStr = f"{amperage:03d}" if amperage < 100 else str(amperage)

        # Build the part number.
        basePart = f"B{intrLetter}{termCode}{poles}{voltageCode}{amperageStr}"
        partNumber = f"{basePart}{suffix}"
        return partNumber

    # H/J Frame Specific Functions
    def getInterruptingLetterHJ(self, voltage, intRating):
        if voltage == 240:
            if intRating == 25:
                return "D"
            elif intRating == 65:
                return "G"
            elif intRating == 100:
                return "J"
            elif intRating == 125:
                return "L"
            elif intRating == 200:
                return "R"
        elif voltage == 480:
            if intRating == 18:
                return "D"
            elif intRating == 35:
                return "G"
            elif intRating == 65:
                return "J"
            elif intRating == 100:
                return "L"
            elif intRating == 200:
                return "R"
        elif voltage == 600:
            if intRating == 14:
                return "D"
            elif intRating == 18:
                return "G"
            elif intRating == 25:
                return "J"
            elif intRating == 50:
                return "L"
            elif intRating == 100:
                return "R"
        return None

    def generateMccbPartNumberHJ(self, frame, termination, poles, amperage, voltage, intRating, tripFunction, grounding, suffix=""):
        """
        Format:
            [Frame] + [interrupting letter] + [termination code] + [poles] + [voltage code] + [amperage_formatted] + [trip_suffix] + [special_suffix]
        Additional rules:
            - termination: "lug" → "L"; "iline" → "A"
            - For H and J frames:
                * Magnetic trip breakers: allowed poles = 2 or 3.
                * Electronic trip breakers: allowed poles must be 3.
            - Allowed amperages depend on both the frame and trip function:
                * For magnetic trip:
                    - H frame allowed amperages: [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 125, 150]
                    - J frame allowed amperages: [150, 175, 200, 225, 250]
                * For electronic trip (only 3-pole allowed):
                    - H frame allowed amperages: [60, 100, 150]
                    - J frame allowed amperages: [250]
            - The voltage code is always "6" (for 240V, 480V, and 600V).
            - For electronic trip breakers, append a trip suffix based on grounding:
                * If grounding is True: suffix = "U44X"
                * If grounding is False: suffix = "U31X"
            - For magnetic trip breakers, no additional trip suffix is added.
        """
        # Validate termination input.
        termUpper = termination.upper()
        if termUpper == "LUG":
            termCode = "L"
        elif termUpper == "ILINE":
            termCode = "A"
        else:
            return f"Invalid termination: {termination}. Use 'LUG' or 'ILINE'."

        # Validate poles.
        if tripFunction.upper() == "ELECTRONIC":
            if poles != 3:
                return f"Electronic trip breakers for {frame} frame must be 3-pole."
        else: # For magnetic trip
            if poles not in [2, 3]:
                return f"Invalid pole count {poles} for {frame} frame magnetic trip breakers; allowed: 2 or 3."

        # Determine the interrupting letter based on voltage and int_rating.
        intrLetter = self.getInterruptingLetterHJ(voltage, intRating)
        if intrLetter is None:
            return f"Invalid interrupting rating {intRating} kA for voltage {voltage}V for {frame} frame."

        # For H frame with magnetic trip, disallow "R"
        if frame == "H" and tripFunction.upper() == "MAGNETIC" and intrLetter == "R":
            return f"Invalid interrupting rating: Magnetic trip H frame breakers cannot have an 'R' interrupting rating."

        # Set allowed amperages and trip function suffix based on trip_function.
        tf = tripFunction.upper()
        if tf == "MAGNETIC":
            if frame == "H":
                allowedAmperages = [15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 125, 150]
            elif frame == "J":
                allowedAmperages = [150, 175, 200, 225, 250]
            tripSuffix = ""
        elif tf == "ELECTRONIC":
            if frame == "H":
                allowedAmperages = [60, 100, 150]
            elif frame == "J":
                allowedAmperages = [250]
            tripSuffix = "U44X" if grounding else "U31X"
        else:
            return f"Invalid trip function '{tripFunction}'. Allowed options: magnetic, electronic."

        if amperage not in allowedAmperages:
            return f"Invalid amperage {amperage}A for {frame} frame with {intrLetter} interrupting rating and {tf} trip. Allowed: {allowedAmperages}"

        # Determine voltage code: "6" for 240V, 480V, or 600V.
        voltageCode = "6"

        # Format amperage.
        amperageStr = f"{amperage:03d}" if amperage < 100 else str(amperage)

        # Build the base part number.
        basePart = f"{frame}{intrLetter}{termCode}{poles}{voltageCode}{amperageStr}"
        partNumber = f"{basePart}{tripSuffix}{suffix}"
        return partNumber

    # Q Frame Specific Functions
    def getInterruptingLetterQ(self, voltage, intRating):
        # Ensure we are dealing with 240V (or lower) systems.
        if voltage > 240:
            return None
        # Use the mapping for 240V:
        mapping = {
            10: "B",
            25: "D",
            65: "G",
            100: "J"
        }
        return mapping.get(intRating, None)

    def generateMccbPartNumberQ(self, termination, poles, amperage, voltage, intRating, suffix=""):
        """
        Generate a Q-frame MCCB part number.
        For Q frame:
            - Only available in thermal magnetic (thus no trip_function input).
            - Only allowed poles are 2 or 3.
            - Only allowed voltage is 240V (or values less than or equal to 240V are treated as 240V).
            - The interrupting rating letter is determined by:
                    240V & 10 kA → "B"
                    240V & 25 kA → "D"
                    240V & 65 kA → "G"
                    240V & 100 kA → "J"
            - Allowed amperages for Q frame are: [70, 80, 90, 100, 110, 125, 150, 175, 200, 225, 250]
            - The voltage code in the part number for Q frame is "2" (instead of "6").
        The final part number format is:
                "Q" + [interrupting letter] + [termination code] + [poles] + [voltage code] + [amperage_formatted] + [special_suffix]
        For example, QBL22070 represents:
                Q-frame, "B" (10 kA), "L" (lug), 2 poles, voltage code "2", amperage "070" (70A).
        """
        # Validate termination input.
        termUpper = termination.upper()
        if termUpper == "LUG":
            termCode = "L"
        elif termUpper == "ILINE":
            termCode = "A"
        else:
            return f"Invalid termination: {termination}. Use 'LUG' or 'ILINE'."

        # Validate poles: only 2 or 3 allowed.
        if poles not in [2, 3]:
            return f"Invalid pole count {poles} for Q frame breakers; allowed: 2 or 3."

        # Validate voltage: Q frame is only available for 240V or lower.
        if voltage > 240:
            return "Invalid voltage for Q frame breakers; only 240V (or below) is allowed."
        # For Q frame, we use 240V as our standard (voltage code "2").
        voltageCode = "2"

        # Determine the interrupting rating letter.
        intrLetter = self.getInterruptingLetterQ(voltage, intRating)
        if intrLetter is None:
            return f"Invalid interrupting rating {intRating} kA for Q frame breakers at {voltage}V."

        # Allowed amperages for Q frame.
        allowedAmperages = [70, 80, 90, 100, 110, 125, 150, 175, 200, 225, 250]
        if amperage not in allowedAmperages:
            return f"Invalid amperage {amperage}A for Q frame breakers. Allowed: {allowedAmperages}"

        # Format amperage. (Since all allowed amperages are >=70, use zero-padding if less than 100.)
        amperageStr = f"{amperage:03d}" if amperage < 100 else str(amperage)

        # Build the base part number.
        basePart = f"Q{intrLetter}{termCode}{poles}{voltageCode}{amperageStr}"

        # Append any special suffix.
        partNumber = f"{basePart}{suffix}"
        return partNumber

    # LL Frame (LA/LH) Specific Functions
    def generateMccbPartNumberLL(self, termination, poles, amperage, voltage, intRating, suffix=""):
        """
        Generate an MCCB part number for LA/LH frame breakers, which are now identified with frame code "LL".
        Rules for LL frame:
            - Only available with magnetic trip (trip_function must be "magnetic", enforced in the driver).
            - Grounding must be False (enforced in the driver).
            - Only 2-pole and 3-pole breakers are allowed.
            - Allowed amperages: [125, 150, 175, 200, 225, 250, 300, 350, 400].
            - Only available for 240V systems.
            - Interrupting rating (in kA) mapping:
                For 240V: 42 → "LAL", 65 → "LHL"
                For 480V: 30 → "LAL", 35 → "LHL"
                For 600V: 22 → "LAL", 25 → "LHL"
            - Uses voltage code "6".
            - Final part number format: "LL" + [interrupting letter] + [termination code] + [poles] + [voltage code] + [amperage] + [special_suffix]
        """
        # Validate termination input.
        termUpper = termination.upper()
        if termUpper == "LUG":
            termCode = "L"
        elif termUpper == "ILINE":
            termCode = ""
        else:
            return f"Invalid termination: {termination}. Only 'lug' is allowed"

        # Validate allowed pole count: only 3p are allowed for our purposes.
        if poles != 3:
            return f"Invalid pole count {poles} for LA/LH (LL) frame breakers; only 3-pole allowed."

        # Validate allowed amperages.
        allowedAmperages = [125, 150, 175, 200, 225, 250, 300, 350, 400]
        if amperage not in allowedAmperages:
            return f"Invalid amperage {amperage}A for LA/LH (LL) frame breakers. Allowed: {allowedAmperages}"

        # Validate voltage: must be one of 240, 480, or 600.
        if voltage not in [240, 480, 600]:
            return f"Invalid voltage {voltage}V for LA/LH (LL) frame breakers; allowed: 240, 480, or 600V."

        # Determine the interrupting letter based on voltage and int_rating.
        if voltage == 240:
            if intRating == 42:
                intrLetter = "A"
            elif intRating == 65:
                intrLetter = "H"
            else:
                return f"Invalid interrupting rating {intRating} kA for 240V LA/LH (LL) frame breaker. Allowed: 42 or 65."
        elif voltage == 480:
            if intRating == 30:
                intrLetter = "A"
            elif intRating == 35:
                intrLetter = "H"
            else:
                return f"Invalid interrupting rating {intRating} kA for 480V LA/LH (LL) frame breaker. Allowed: 30 or 35."
        elif voltage == 600:
            if intRating == 22:
                intrLetter = "A"
            elif intRating == 25:
                intrLetter = "H"
            else:
                return f"Invalid interrupting rating {intRating} kA for 600V LA/LH (LL) frame breaker. Allowed: 22 or 25."

        # Use voltage code "2" for LA/LH breakers.
        voltageCode = "6"
        # Amperage as string (no zero padding needed since minimum is 125).
        amperageStr = str(amperage)

        # Build the part number.
        basePart = f"L{intrLetter}{termCode}{poles}{voltageCode}{amperageStr}"
        partNumber = f"{basePart}{suffix}"
        return partNumber

    # L Frame Electronic Trip Specific Functions
    def getInterruptingLetterLElectronic(self, voltage, intRating):
        if voltage == 240:
            mapping = {65: "G", 100: "J", 125: "L", 200: "R"}
        elif voltage == 480:
            mapping = {35: "G", 65: "J", 100: "L", 200: "R"}
        elif voltage == 600:
            mapping = {18: "G", 25: "J", 50: "L", 100: "R"}
        else:
            return None
        return mapping.get(intRating, None)

    def generateMccbPartNumberLElectronic(self, termination, poles, amperage, voltage, intRating, grounding, suffix=""):
        """
        Generate an MCCB part number for L frame electronic trip breakers.
        Rules for L electronic:
            - Only allowed pole count is 3.
            - Allowed amperages depend on grounding:
                    * Without grounding (grounding == False): allowed amperages = [250, 400, 600]
                    * With grounding (grounding == True): allowed amperages = [400, 600]
            - Only 240V, 480V, or 600V are allowed.
            - Interrupting letter is determined by:
                    For 240V: 65 → G, 100 → J, 125 → L, 200 → R
                    For 480V: 35 → G, 65 → J, 100 → L, 200 → R
                    For 600V: 18 → G, 25 → J, 50 → L, 100 → R
            - Termination: if termination is "lug" then code is "L"; if "iline" then blank.
            - Voltage code is "6".
            - For electronic trip breakers, append a trip suffix based on grounding:
                    * If grounding is True: suffix = "U44X"
                    * If grounding is False: suffix = "U31X"
        """
        # Validate termination.
        termUpper = termination.upper()
        if termUpper == "LUG":
            termCode = "L"
        elif termUpper == "ILINE":
            termCode = ""
        else:
            return f"Invalid termination: {termination}. Only 'lug' is allowed for L frame electronic breakers."

        # Validate pole count.
        if poles != 3:
            return f"Invalid pole count {poles} for L frame electronic breakers; only 3-pole breakers are allowed."

        # Validate voltage.
        if voltage not in [240, 480, 600]:
            return f"Invalid voltage {voltage}V for L frame electronic breakers; allowed: 240, 480, or 600V."

        # Determine interrupting letter.
        if voltage == 240:
            mapping = {65: "G", 100: "J", 125: "L", 200: "R"}
        elif voltage == 480:
            mapping = {35: "G", 65: "J", 100: "L", 200: "R"}
        elif voltage == 600:
            mapping = {18: "G", 25: "J", 50: "L", 100: "R"}
        intrLetter = mapping.get(intRating, None)
        if intrLetter is None:
            return f"Invalid interrupting rating {intRating} kA for {voltage}V L frame electronic breakers."

        # Set allowed amperages based on grounding.
        if bool(grounding):
            allowedAmperages = [400, 600]
        else:
            allowedAmperages = [250, 400, 600]
        if amperage not in allowedAmperages:
            return f"Invalid amperage {amperage}A for L frame electronic breakers with grounding={grounding}. Allowed: {allowedAmperages}"

        # Use voltage code "6" (same as other L frames).
        voltageCode = "6"

        # Format amperage.
        amperageStr = str(amperage)  # No 0 padding needed.

        # Determine trip suffix based on grounding (explicitly check boolean).
        tripSuffix = "U44X" if bool(grounding) else "U31X"

        # Build the part number.
        basePart = f"L{intrLetter}{termCode}{poles}{voltageCode}{amperageStr}"
        partNumber = f"{basePart}{tripSuffix}{suffix}"
        return partNumber

    # M Frame Specific Functions
    def getInterruptingLetterM(self, voltage, intRating):
        if voltage == 240:
            if intRating == 65:
                return "G"
            elif intRating == 100:
                return "J"
            else:
                return None
        elif voltage == 480:
            if intRating == 35:
                return "G"
            elif intRating == 65:
                return "J"
            else:
                return None
        elif voltage == 600:
            if intRating == 18:
                return "G"
            elif intRating == 25:
                return "J"
            else:
                return None
        else:
            return None

    def generateMccbPartNumberM(self, termination, poles, amperage, voltage, intRating, grounding=False, suffix=""):
        """
        Generate an MCCB part number for M frame breakers.
        Rules for M frame (electronic trip only):
            - Allowed poles: 2 or 3.
            - Allowed amperages: only 400A or 600A.
            - The interrupting letter is determined as follows:
                For 240V: 65 kA → "G", 100 kA → "J"
                For 480V: 35 kA → "G", 65 kA → "J"
                For 600V: 18 kA → "G", 25 kA → "J"
            - The amperage code is "4" if amperage is 400 and "8" if amperage is 600.
            - The voltage code is fixed as "6".
            - Final format: "M" + [interrupting letter] + [termination code] + [poles] + [voltage code] + [amperage code] + "00" + [special_suffix]
            - Grounding must always be False for M frame breakers.
        """
        # Enforce grounding must be False.
        if grounding:
            return "Invalid configuration: M frame breakers must have grounding set to False."
        # Validate termination input.
        termUpper = termination.upper()
        if termUpper == "LUG":
            termCode = "L"
        elif termUpper == "ILINE":
            termCode = "A"
        else:
            return f"Invalid termination: {termination}. Use 'lug' or 'iline'."
        # Validate poles.
        if poles not in [2, 3]:
            return f"Invalid pole count {poles} for M frame breakers; allowed: 2 or 3."
        # Validate amperage.
        if amperage not in [400, 600]:
            return f"Invalid amperage {amperage}A for M frame breakers; allowed: 400 or 600."
        # Validate voltage.
        if voltage not in [240, 480, 600]:
            return f"Invalid voltage {voltage}V for M frame breakers; allowed: 240, 480, or 600V."
        # Determine the interrupting letter.
        intrLetter = self.getInterruptingLetterM(voltage, intRating)
        if intrLetter is None:
            return f"Invalid interrupting rating {intRating} kA for voltage {voltage}V for M frame breakers."
        # Determine amperage code.
        ampCode = "4" if amperage == 400 else "8"
        voltageCode = "6"
        basePart = f"M{intrLetter}{termCode}{poles}{voltageCode}{ampCode}00"
        partNumber = f"{basePart}{suffix}"
        return partNumber

        # P Frame Specific Functions
    
    # P Frame Specific Functions
    def getInterruptingLetterP(self, voltage, intRating):
        if voltage == 240:
            mapping = {65: "G", 100: "J"}
        elif voltage == 480:
            mapping = {35: "G", 65: "J", 50: "K", 100: "L"}
        elif voltage == 600:
            mapping = {18: "G", 25: "J", 50: "K"}
        else:
            return None
        return mapping.get(intRating, None)

    def generateMccbPartNumberP(self, termination, poles, amperage, voltage, intRating, grounding, suffix=""):
        # Validate termination.
        termUpper = termination.upper()
        if termUpper == "LUG":
            termCode = "L"
        elif termUpper == "ILINE":
            termCode = "A"
        else:
            return f"Invalid termination: {termination}. Use 'lug' or 'iline'."

        # Only 3‑pole allowed
        if poles != 3:
            return f"Invalid pole count {poles} for P frame breakers; allowed: 3."
        # Voltage must be one of these
        if voltage not in [240, 480, 600]:
            return f"Invalid voltage {voltage}V for P frame breakers; allowed: 240, 480, or 600V."

        # Determine interrupting letter
        intrLetter = self.getInterruptingLetterP(voltage, intRating)
        if intrLetter is None:
            return f"Invalid interrupting rating {intRating} kA for P frame breakers at {voltage}V."

        # "L" only valid at 480V
        if intrLetter == "L" and voltage != 480:
            return f"Interrupting rating 'L' is only valid for 480V P frame breakers."

        # Allowed amperages
        allowedAmps = [250, 400, 600, 800, 1000, 1200]
        if amperage not in allowedAmps:
            return f"Invalid amperage {amperage}A for P frame breakers. Allowed: {allowedAmps}"

        # Two‑digit amps (250, 400)
        if amperage in [250, 400]:
            voltageCode  = "4" if (intrLetter == "L" and voltage == 480) else "6"
            ampCode      = f"{int(amperage/10):03d}"            # zero‑pad to 3 digits (025, 040)
            basePart     = f"P{intrLetter}{termCode}{poles}{voltageCode}{ampCode}"
            tripSuffix   = "U44A" if grounding else "U31A"

        # 600A and above
        else:
            voltageCode  = "4" if (intrLetter == "L" and voltage == 480) else "6"
            ampDiv       = amperage // 10
            ampCode      = f"{ampDiv:03d}"                      # 600→060, 800→080, 1000→100, etc.
            basePart     = f"P{intrLetter}{termCode}{poles}{voltageCode}{ampCode}"
            tripSuffix   = "U44A" if grounding else ""

        return f"{basePart}{tripSuffix}{suffix}"

    # R Frame Specific Functions
    def getInterruptingLetterR(self, voltage, intRating):
        if voltage == 240:
            mapping = {65: "G", 100: "J", 125: "L"}
        elif voltage == 480:
            mapping = {35: "G", 65: "J", 100: "L"}
        elif voltage == 600:
            mapping = {18: "G", 25: "J", 50: "L", 65: "K"}
        else:
            return None
        return mapping.get(intRating, None)

    def generateMccbPartNumberR(self, termination, poles, amperage, voltage, intRating, grounding, suffix="", tripFunction=None):
        """
        Rules for R frame (electronic trip only):
            - Only allows electronic trip (tripFunction must be 'electronic')
            - Only allowed poles: 3
            - Termination: "lug" → "F", "iline" → "A"
            - ILINE only supports 1000A and 1200A, with catalog suffix 'C'
            - Interrupting letter based on voltage and kA mapping
            - Amperage code: 3-digit zero-padded version (e.g., 1000 → "100")
            - Suffix rules:
                * If grounding: always append "U44A"
                * If ungrounded and amperage in [600,800,1000,3000]: append "U31A"
        """
        # 1) Trip‑function guard
        if tripFunction is None or tripFunction.upper() != "ELECTRONIC":
            return "R frame breakers only allow electronic trip."

        # 2) Termination + catalogSuffix
        termUpper = termination.upper()
        if termUpper == "LUG":
            termCode = "F"
            catalogSuffix = ""
        elif termUpper == "ILINE":
            termCode = "A"
            # restrict ILINE to 1000A/1200A
            if amperage not in [1000, 1200]:
                return f"Invalid amperage {amperage}A for ILINE R frame breakers; allowed: 1000 or 1200."
            catalogSuffix = "C"
        else:
            return f"Invalid termination: {termination}. Use 'lug' or 'iline'."

        # 3) Poles check
        if poles != 3:
            return "R frame breakers only allow 3-pole configurations."

        # 4) Voltage check
        if voltage not in [240, 480, 600]:
            return f"Invalid voltage {voltage}V for R frame breakers; allowed: 240, 480, or 600V."

        # 5) Interrupting letter
        intrLetter = self.getInterruptingLetterR(voltage, intRating)
        if intrLetter is None:
            return f"Invalid interrupting rating {intRating} kA for R frame at {voltage}V."

        # 6) Non-ILINE amperage validation
        if termUpper != "ILINE":
            allowedAmps = [600, 800, 1000, 1200, 1600, 2000, 2500, 3000]
            if amperage not in allowedAmps:
                return f"Invalid amperage {amperage}A for R frame breakers. Allowed: {allowedAmps}"

        # 7) Build base part
        amperageCode = f"{int(amperage/10):03d}"
        voltageCode  = "6"
        basePart     = f"R{intrLetter}{termCode}{poles}{voltageCode}{amperageCode}"

        # 8) Trip suffix
        if grounding:
            tripSuffix = "U44A"
        else:
            # ILINE ungrounded always gets U31A; Lug follows original size list
            if termUpper == "ILINE" or amperage in [600, 800, 1000, 3000]:
                tripSuffix = "U31A"
            else:
                tripSuffix = ""

        # 9) Final part number assembly
        return f"{basePart}{catalogSuffix}{tripSuffix}{suffix}"

         
    # Enclosure Lookup Function
    def mccbEnclosure(self, partNumber, poles, amperage, enclosureType):
        """
        Given a part number, poles, amperage, and enclosure_type (Flush, Surface, NEMA 3R),
        return the correct enclosure part number or None if not applicable.
        """
        if not enclosureType or enclosureType.upper() == "NONE":
            return None

        enclosureType = enclosureType.upper()
        first3 = partNumber[:3]

        def matchType(options):
            if enclosureType == "FLUSH" and options.get("FLUSH"): return options["FLUSH"]
            if enclosureType == "SURFACE" and options.get("SURFACE"): return options["SURFACE"]
            if enclosureType in "NEMA3R" and options.get("NEMA3R"): return options["NEMA3R"]
            return None

        enclosureMap = {
            "BDL": {"FLUSH": "B125F", "SURFACE": "B125S", "NEMA3R": "B125RB"},
            "BGL": {"FLUSH": "B125F", "SURFACE": "B125S", "NEMA3R": "B125RB"},
            "BJL": {"FLUSH": "B125F", "SURFACE": "B125S", "NEMA3R": "B125RB"},
            "BKL": {"FLUSH": "B125F", "SURFACE": "B125S", "NEMA3R": "B125RB"},
            "HDL": {**({"FLUSH": "H150F", "SURFACE": "H150S", "NEMA3R": "H150R"} if poles == 2 else {})},
            "HGL": {**({"FLUSH": "H150F", "SURFACE": "H150S", "NEMA3R": "H150R"} if poles == 2 else {"FLUSH": "J250F", "SURFACE": "J250S", "NEMA3R": "J250R"})},
            "HJL": {"FLUSH": "J250F", "SURFACE": "J250S", "NEMA3R": "J250R"},
            "HLL": {"FLUSH": "J250F", "SURFACE": "J250S", "NEMA3R": "J250R"},
            "JDL": {"FLUSH": "J250F", "SURFACE": "J250S", "NEMA3R": "J250R"},
            "JGL": {"FLUSH": "J250F", "SURFACE": "J250S", "NEMA3R": "J250R"},
            "JJL": {"FLUSH": "J250F", "SURFACE": "J250S", "NEMA3R": "J250R"},
            "JLL": {"FLUSH": "J250F", "SURFACE": "J250S", "NEMA3R": "J250R"},
            "LDL": {"NEMA3R": "L600AWK"},
            "LJL": {"NEMA3R": "L600AWK"},
            "LGL": {"NEMA3R": "L600AWKMC"},
            "LLL": {"NEMA3R": "L600AWKMC"},
            "LRL": {"NEMA3R": "L600AWKMC"},
            "QDL": {"SURFACE": "Q23225NS", "NEMA3R": "Q23225NRB"},
            "QGL": {"SURFACE": "Q23225NS", "NEMA3R": "Q23225NRB"},
            "QJL": {"SURFACE": "Q23225NS", "NEMA3R": "Q23225NRB"},
            "QBL": {"SURFACE": "Q23225NS", "NEMA3R": "Q23225NRB"},
            "MGL": {"SURFACE": "M800S", "NEMA3R": "M800R"},
            "MJL": {"SURFACE": "M800S", "NEMA3R": "M800R"},
            "PGL": {"2": {"SURFACE": "M800S", "NEMA3R": "M800R"}, "3": {"SURFACE": "P1200S", "NEMA3R": "P1200R"}}[str(poles)],
            "PJL": {"2": {"SURFACE": "M800S", "NEMA3R": "M800R"}, "3": {"SURFACE": "P1200S", "NEMA3R": "P1200R"}}[str(poles)],
            "PKL": {"2": {"SURFACE": "M800S", "NEMA3R": "M800R"}, "3": {"SURFACE": "P1200S", "NEMA3R": "P1200R"}}[str(poles)],
            "PLL": {"2": {"SURFACE": "M800S", "NEMA3R": "M800R"}, "3": {"SURFACE": "P1200S", "NEMA3R": "P1200R"}}[str(poles)],
            "LAL": {"FLUSH": "LA400F", "SURFACE": "LA400S", "NEMA3R": "LA400R"},
            "LHL": {"FLUSH": "LA400F", "SURFACE": "LA400S", "NEMA3R": "LA400R"},
        }

        return matchType(enclosureMap.get(first3, {}))

# End MCCB ^

