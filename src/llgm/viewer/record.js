"use strict";
async function openRecord() {
  const preview = document.getElementById("record-content");
  try {
    const response = await fetch("api/file" + location.search, {
      cache: "no-store",
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Unable to open record");
    }
    preview.textContent = await response.text();
  } catch (error) {
    preview.textContent = error.message;
  }
}
openRecord();
