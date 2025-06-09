from agent import app

initial_state = {
    "project_type": "react-native",
    "goal": "",
    "structure": {
        "src": {
            "components": {},
            "screens": {
                "HomeScreen.tsx": "",
                "DetailsScreen.tsx": ""
            },
            "services": {
                "supabase.ts": ""
            },
            "navigation": {
                "AppNavigator.tsx": ""
            },
            "App.tsx": ""
        },
        "package.json": "",
        "tsconfig.json": "",
        "babel.config.js": ""
    }
}

app.invoke(initial_state)
