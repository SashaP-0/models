

    /*            --------------------------------------------------
                    :: Initialize mega menu
                    -------------------------------------------------- */
    jQuery(function ($) {
        $('.droopmenu-navbar').droopmenu({
            dmArrow: true,
            dmArrowDirection: 'dmarrowup'
        });
    });
    $(document).ready(function () {

        $(".fa-search").click(function () {
            $(".search-box").toggle();
            $("input[type='text'].search").focus();
        });
    });
