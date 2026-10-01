document.addEventListener(
    "DOMContentLoaded",
    () => {
        document
            .querySelectorAll("table")
            .forEach((table) => {
                const headers =
                    table.querySelectorAll(
                        "thead th"
                    );

                headers.forEach(
                    (
                        header,
                        columnIndex
                    ) => {
                        header.classList.add(
                            "sortable"
                        );

                        header.setAttribute(
                            "role",
                            "button"
                        );

                        header.setAttribute(
                            "tabindex",
                            "0"
                        );

                        let ascending =
                            true;

                        const sort =
                            () => {
                                const tbody =
                                    table.querySelector(
                                        "tbody"
                                    );

                                if (!tbody) {
                                    return;
                                }

                                const rows =
                                    Array.from(
                                        tbody.querySelectorAll(
                                            "tr"
                                        )
                                    ).filter(
                                        (
                                            row
                                        ) =>
                                            !row.querySelector(
                                                ".empty"
                                            )
                                    );

                                rows.sort(
                                    (
                                        a,
                                        b
                                    ) => {
                                        const aCell =
                                            a.children[
                                                columnIndex
                                            ];

                                        const bCell =
                                            b.children[
                                                columnIndex
                                            ];

                                        const aValue =
                                            aCell?.dataset
                                                .sortValue ??
                                            aCell?.textContent.trim() ??
                                            "";

                                        const bValue =
                                            bCell?.dataset
                                                .sortValue ??
                                            bCell?.textContent.trim() ??
                                            "";

                                        const aNumber =
                                            parseFloat(
                                                aValue
                                                    .replace(
                                                        /\s/g,
                                                        ""
                                                    )
                                                    .replace(
                                                        ",",
                                                        "."
                                                    )
                                                    .replace(
                                                        "%",
                                                        ""
                                                    )
                                            );

                                        const bNumber =
                                            parseFloat(
                                                bValue
                                                    .replace(
                                                        /\s/g,
                                                        ""
                                                    )
                                                    .replace(
                                                        ",",
                                                        "."
                                                    )
                                                    .replace(
                                                        "%",
                                                        ""
                                                    )
                                            );

                                        if (
                                            !Number.isNaN(
                                                aNumber
                                            ) &&
                                            !Number.isNaN(
                                                bNumber
                                            )
                                        ) {
                                            return ascending
                                                ? aNumber -
                                                      bNumber
                                                : bNumber -
                                                      aNumber;
                                        }

                                        return ascending
                                            ? aValue.localeCompare(
                                                  bValue,
                                                  "sv"
                                              )
                                            : bValue.localeCompare(
                                                  aValue,
                                                  "sv"
                                              );
                                    }
                                );

                                rows.forEach(
                                    (row) =>
                                        tbody.appendChild(
                                            row
                                        )
                                );

                                headers.forEach(
                                    (
                                        item
                                    ) => {
                                        item.classList.remove(
                                            "sort-ascending",
                                            "sort-descending"
                                        );
                                    }
                                );

                                header.classList.add(
                                    ascending
                                        ? "sort-ascending"
                                        : "sort-descending"
                                );

                                ascending =
                                    !ascending;
                            };

                        header.addEventListener(
                            "click",
                            sort
                        );

                        header.addEventListener(
                            "keydown",
                            (event) => {
                                if (
                                    event.key ===
                                        "Enter" ||
                                    event.key ===
                                        " "
                                ) {
                                    event.preventDefault();
                                    sort();
                                }
                            }
                        );
                    }
                );
            });
    }
);
