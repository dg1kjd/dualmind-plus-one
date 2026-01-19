// Title: DualMind+1 Moshi Edition Client Application
// Description: This is the main React application component for the DualMind+1 Moshi Edition client, managing user interface and audio interactions.
// Author: Jens David
// Copyright: 2026 Jens David Consulting
// License: MIT

import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import "./index.css";
import { Queue } from "./pages/Queue/Queue";

const router = createBrowserRouter([
  {
    path: "/",
    element: <Queue />,
  },
]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
    <RouterProvider router={router}/>
);
