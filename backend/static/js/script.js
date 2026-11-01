
console.log("KrishiMitra AI loaded successfully!");

// ==========================================
// 1. GLOBAL LANGUAGE SETTINGS
// ==========================================

const languageToggle = document.getElementById("languageToggle");
const stateSelect = document.getElementById("state");
const districtSelect = document.getElementById("district");

let currentLanguage = localStorage.getItem("krishiLanguage") || "en";

// ==========================================
// 2. RAJASTHAN DISTRICT HINDI TRANSLATIONS
// ==========================================

const hindiDistricts = {
    "Ajmer": "अजमेर",
    "Alwar": "अलवर",
    "Banswara": "बांसवाड़ा",
    "Baran": "बारां",
    "Barmer": "बाड़मेर",
    "Bharatpur": "भरतपुर",
    "Bhilwara": "भीलवाड़ा",
    "Bikaner": "बीकानेर",
    "Bundi": "बूंदी",
    "Chittaurgarh": "चित्तौड़गढ़",
    "Churu": "चूरू",
    "Dausa": "दौसा",
    "Dholpur": "धौलपुर",
    "Dungarpur": "डूंगरपुर",
    "Ganganagar": "गंगानगर",
    "Hanumangarh": "हनुमानगढ़",
    "Jaipur": "जयपुर",
    "Jaisalmer": "जैसलमेर",
    "Jalore": "जालौर",
    "Jhalawar": "झालावाड़",
    "Jhunjhunun": "झुंझुनूं",
    "Jodhpur": "जोधपुर",
    "Karauli": "करौली",
    "Kota": "कोटा",
    "Nagaur": "नागौर",
    "Pali": "पाली",
    "Pratapgarh": "प्रतापगढ़",
    "Rajsamand": "राजसमंद",
    "Sawai Madhopur": "सवाई माधोपुर",
    "Sikar": "सीकर",
    "Sirohi": "सिरोही",
    "Tonk": "टोंक",
    "Udaipur": "उदयपुर"
};

// ==========================================
// 3. DISTRICT PLACEHOLDER
// ==========================================

function setDistrictPlaceholder(en, hi) {
    if (!districtSelect) return;

    districtSelect.innerHTML = "";

    const option = document.createElement("option");

    option.value = "";
    option.dataset.en = en;
    option.dataset.hi = hi;

    option.textContent = currentLanguage === "hi" ? hi : en;

    districtSelect.appendChild(option);
}

// ==========================================
// 4. DISTRICT LANGUAGE TRANSLATION
// ==========================================

function updateDistrictLanguage() {
    if (!districtSelect) return;

    Array.from(districtSelect.options).forEach(option => {

        if (option.dataset.en && option.dataset.hi) {

            const translatedText =
                currentLanguage === "hi"
                    ? option.dataset.hi
                    : option.dataset.en;

            option.textContent = translatedText;
        }
    });
}

// ==========================================
// 5. MAIN LANGUAGE SWITCH FUNCTION
// ==========================================

function changeLanguage(language) {

    currentLanguage = language;

    document.documentElement.lang = language;

    // Translate all normal page elements
    document.querySelectorAll("[data-en][data-hi]").forEach(element => {

        element.textContent = element.getAttribute("data-" + language);

    });

    // Translate dynamically created district options
    updateDistrictLanguage();

    // Update language toggle button
    if (languageToggle) {

        languageToggle.textContent =
            language === "en" ? "हिन्दी" : "English";

    }

    localStorage.setItem("krishiLanguage", language);
}

// ==========================================
// 6. LANGUAGE BUTTON EVENT
// ==========================================

if (languageToggle) {

    languageToggle.addEventListener("click", function () {

        const newLanguage =
            currentLanguage === "en" ? "hi" : "en";

        changeLanguage(newLanguage);

    });

}

// ==========================================
// 7. LOAD SAVED LANGUAGE
// ==========================================

changeLanguage(currentLanguage);

// ==========================================
// 8. LOAD DISTRICTS FROM FLASK API
// ==========================================

if (stateSelect && districtSelect) {

    stateSelect.addEventListener("change", async function () {

        const selectedState = stateSelect.value;

        districtSelect.disabled = true;

        if (!selectedState) {

            setDistrictPlaceholder(
                "Select your state first",
                "पहले अपना राज्य चुनें"
            );

            return;
        }

        setDistrictPlaceholder(
            "Loading districts...",
            "जिले लोड हो रहे हैं..."
        );

        try {

            const response = await fetch(
                `/api/districts?state=${encodeURIComponent(selectedState)}`
            );

            if (!response.ok) {
                throw new Error("District API request failed");
            }

            const districts = await response.json();

            if (!Array.isArray(districts) || districts.length === 0) {

                setDistrictPlaceholder(
                    "No districts available",
                    "कोई जिला उपलब्ध नहीं है"
                );

                return;
            }

            setDistrictPlaceholder(
                "Select District",
                "जिला चुनें"
            );

            districts.forEach(district => {

                const option = document.createElement("option");

                // Keep English value for Flask and dataset matching
                option.value = district;

                option.dataset.en = district;

                option.dataset.hi =
                    hindiDistricts[district] || district;

                districtSelect.appendChild(option);

            });

            // Apply the currently selected language
            updateDistrictLanguage();

            districtSelect.disabled = false;

        } catch (error) {

            console.error("District loading error:", error);

            setDistrictPlaceholder(
                "Unable to load districts. Try again.",
                "जिले लोड नहीं हो पाए। पुनः प्रयास करें।"
            );

        }

    });

}

// ==========================================
// 9. END OF KRISHIMITRA AI SCRIPT
// ==========================================
